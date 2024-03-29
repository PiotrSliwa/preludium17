import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Callable, Dict, Iterator, Tuple

from scipy.sparse import csr_matrix
from sklearn.feature_extraction import DictVectorizer

from timelines import Timeline


class FeatureClass(Enum):
    POSITIVE = 1
    NEGATIVE = 0


FeatureName = str
FeatureDict = Dict[FeatureName, float]
Dicterizer = Callable[[Timeline], FeatureDict]


class TimelineDataset:
    __x: List[Timeline]
    __y: List[FeatureClass]
    __test: List[bool]

    @dataclass
    class Metrics:
        training_datasets: int
        test_datasets: int
        test_to_training_ratio: float
        training_positive_classes: int
        training_negative_classes: int
        training_class_ratio: float
        test_positive_classes: int
        test_negative_classes: int
        test_class_ratio: float

    def __init__(self, x: List[Timeline] = None, y: List[FeatureClass] = None, test: List[bool] = None):
        self.__x = [] if x is None else x
        self.__y = [] if y is None else y
        self.__test = [] if test is None else test
        if len(self.__x) != len(self.__y) or len(self.__x) != len(self.__test):
            raise Exception(f'len({x}) != len({y}) or len({x}) != len({test})')

    def __add__(self, other):
        x = self.__x + other.__x
        y = self.__y + other.__y
        test = self.__test + other.__test
        return TimelineDataset(x, y, test)

    def feature_dicts(self, dicterizer: Dicterizer) -> List[FeatureDict]:
        return list(map(lambda x: dicterizer(x), self.__x))

    def feature_classes(self, shuffle: bool = False) -> List[FeatureClass]:
        result = self.__y.copy()
        if shuffle:
            random.shuffle(result)
        return result

    def test_indices(self) -> List[int]:
        return [i for i, v in enumerate(self.__test) if v]

    def metrics(self) -> Metrics:
        training_positive_classes = 0
        test_positive_classes = 0
        training_negative_classes = 0
        test_negative_classes = 0
        training_datasets = 0
        test_datasets = 0
        for i, y in enumerate(self.__y):
            if self.__test[i]:
                test_datasets += 1
                if y == FeatureClass.POSITIVE:
                    test_positive_classes += 1
                else:
                    test_negative_classes += 1
            else:
                training_datasets += 1
                if y == FeatureClass.POSITIVE:
                    training_positive_classes += 1
                else:
                    training_negative_classes += 1
        return TimelineDataset.Metrics(
            training_datasets=training_datasets,
            test_datasets=test_datasets,
            test_to_training_ratio=test_datasets / (training_datasets + test_datasets),
            training_positive_classes=training_positive_classes,
            training_negative_classes=training_negative_classes,
            training_class_ratio=training_positive_classes / (training_positive_classes + training_negative_classes),
            test_positive_classes=test_positive_classes,
            test_negative_classes=test_negative_classes,
            test_class_ratio=test_positive_classes / (test_positive_classes + test_negative_classes),
        )


TrainIndices = List[int]
TestIndices = List[int]
Split = Tuple[TrainIndices, TestIndices]


@dataclass(frozen=True)
class SklearnDataset:
    X: csr_matrix
    y: List[int]
    splits: List[Split]


def timeline_to_sklearn_dataset(dataset: TimelineDataset, dicterizer: Dicterizer, shuffle_classes: bool = False) -> SklearnDataset:
    feature_dicts = dataset.feature_dicts(dicterizer)
    feature_classes = dataset.feature_classes(shuffle_classes)
    vectorizer = DictVectorizer()
    X = vectorizer.fit_transform(feature_dicts)
    y = list(map(lambda x: x.value, feature_classes))
    test_indices = dataset.test_indices()
    train_indices = [i for i in range(len(feature_dicts)) if i not in test_indices]
    return SklearnDataset(X, y, [(train_indices, test_indices)])
