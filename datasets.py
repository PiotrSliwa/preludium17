from dataclasses import dataclass
from enum import Enum
from typing import List, Callable, Dict, Generic, TypeVar

from scipy.sparse import csr_matrix
from sklearn.feature_extraction import DictVectorizer

from database import Timeline


class FeatureClass(Enum):
    POSITIVE = 1
    NEGATIVE = 0


FeatureName = str
FeatureDict = Dict[FeatureName, float]
Dicterizer = Callable[[Timeline], FeatureDict]


class TimelineDataset:
    __x: List[Timeline]
    __y: List[FeatureClass]

    def __init__(self, x: List[Timeline] = None, y: List[FeatureClass] = None):
        self.__x = [] if x is None else x
        self.__y = [] if y is None else y
        if len(self.__x) != len(self.__y):
            raise Exception(f'${x} != ${y}')

    def __add__(self, other):
        x = self.__x + other.__x
        y = self.__y + other.__y
        return TimelineDataset(x, y)

    def feature_dicts(self, dicterizer: Dicterizer) -> List[FeatureDict]:
        return list(map(lambda x: dicterizer(x), self.__x))

    def feature_classes(self) -> List[FeatureClass]:
        return self.__y.copy()


T = TypeVar('T')


@dataclass
class DatasetSplit(Generic[T]):
    train: List[T]
    test: List[T]

    def __add__(self, other):
        train = self.train + other.train
        test = self.test + other.test
        return DatasetSplit(train, test)

    def __len__(self):
        return len(self.train) + len(self.test)


@dataclass(frozen=True)
class SklearnDataset:
    X: csr_matrix
    y: List[int]


def timeline_to_sklearn_dataset(dataset: TimelineDataset, dicterizer: Dicterizer) -> SklearnDataset:
    feature_dicts = dataset.feature_dicts(dicterizer)
    feature_classes = dataset.feature_classes()
    vectorizer = DictVectorizer()
    X = vectorizer.fit_transform(feature_dicts)
    y = list(map(lambda x: x.value, feature_classes))
    return SklearnDataset(X, y)
