from datetime import datetime
import numpy as np

import pytest

from database import Reference, Timeline
from datasets import TimelineDataset, FeatureClass, timeline_to_sklearn_dataset, DatasetSplit
from dicterizers import counting_dicterizer

now = datetime.now()


def test_timeline_dataset_init_fail():
    with pytest.raises(Exception) as e:
        TimelineDataset([], [FeatureClass.POSITIVE])


def test_timeline_dataset_init():
    timeline: Timeline = [Reference('Reference', datetime.now())]
    dataset = TimelineDataset([timeline], [FeatureClass.POSITIVE])
    assert dataset.feature_dicts(counting_dicterizer) == [{'Reference': 1}]


def test_timeline_dataset_add_operator():
    dataset_a = TimelineDataset([[Reference('Reference_A', datetime.now())]], [FeatureClass.POSITIVE])
    dataset_b = TimelineDataset([[Reference('Reference_B', datetime.now())]], [FeatureClass.POSITIVE])
    dataset = dataset_a + dataset_b
    assert dataset.feature_dicts(counting_dicterizer) == [{'Reference_A': 1}, {'Reference_B': 1}]


def test_timeline_to_sklearn_dataset():
    timeline: Timeline = [Reference('Reference_A', now), Reference('Reference_A', now), Reference('Reference_B', now)]
    timeline_dataset = TimelineDataset([timeline], [FeatureClass.POSITIVE])
    sklearn_dataset = timeline_to_sklearn_dataset(timeline_dataset, counting_dicterizer)
    assert np.all(sklearn_dataset.X.toarray() == [2, 1])
    assert sklearn_dataset.y == [1]


def test_dataset_split():
    split_a = DatasetSplit([1], [2, 3])
    split_b = DatasetSplit([4, 5], [6])
    split = split_a + split_b
    assert len(split) == 6
    assert split.train == [1, 4, 5]
    assert split.test == [2, 3, 6]
