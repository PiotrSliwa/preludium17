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
        positive_classes: int
        negative_classes: int
        positive_to_negative_ratio: float
        training_datasets: int
        test_datasets: int
        training_to_test_ratio: float

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

    def feature_classes(self) -> List[FeatureClass]:
        return self.__y.copy()

    def test_indices(self) -> List[int]:
        return [i for i, v in enumerate(self.__test) if v]

    def metrics(self) -> Metrics:
        positive_classes = sum((int(y == FeatureClass.POSITIVE) for y in self.__y))
        negative_classes = sum((int(y == FeatureClass.NEGATIVE) for y in self.__y))
        training_datasets = sum(1 for test in self.__test if not test)
        test_datasets = sum(1 for test in self.__test if test)
        return TimelineDataset.Metrics(
            positive_classes=positive_classes,
            negative_classes=negative_classes,
            positive_to_negative_ratio=positive_classes / negative_classes,
            training_datasets=training_datasets,
            test_datasets=test_datasets,
            training_to_test_ratio=training_datasets / test_datasets
        )


TrainIndices = List[int]
TestIndices = List[int]
Split = Tuple[TrainIndices, TestIndices]


@dataclass(frozen=True)
class SklearnDataset:
    X: csr_matrix
    y: List[int]
    splits: List[Split]


def timeline_to_sklearn_dataset(dataset: TimelineDataset, dicterizer: Dicterizer) -> SklearnDataset:
    feature_dicts = dataset.feature_dicts(dicterizer)
    feature_classes = dataset.feature_classes()
    vectorizer = DictVectorizer()
    X = vectorizer.fit_transform(feature_dicts)
    y = list(map(lambda x: x.value, feature_classes))
    test_indices = dataset.test_indices()
    train_indices = [i for i in range(len(feature_dicts)) if i not in test_indices]
    return SklearnDataset(X, y, [(train_indices, test_indices)])