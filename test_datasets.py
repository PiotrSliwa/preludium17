from datetime import datetime
import numpy as np

import pytest

from timelines import Timeline, Reference
from datasets import TimelineDataset, FeatureClass, timeline_to_sklearn_dataset
from dicterizers import counting_dicterizer

now = datetime.now()


def test_timeline_dataset_init_fail():
    with pytest.raises(Exception) as e:
        TimelineDataset([], [FeatureClass.POSITIVE], [])
    with pytest.raises(Exception) as e:
        TimelineDataset([], [], [True])
    with pytest.raises(Exception) as e:
        TimelineDataset([[Reference('Reference', now)]], [], [])


def test_timeline_dataset_init():
    timeline: Timeline = [Reference('Reference', now)]
    dataset = TimelineDataset([timeline], [FeatureClass.POSITIVE], [False])
    assert dataset.feature_dicts(counting_dicterizer) == [{'Reference': 1}]


def test_timeline_dataset_add_operator():
    dataset_a = TimelineDataset([[Reference('Reference_A', now)]], [FeatureClass.POSITIVE], [False])
    dataset_b = TimelineDataset([[Reference('Reference_B', now)]], [FeatureClass.POSITIVE], [False])
    dataset = dataset_a + dataset_b
    assert dataset.feature_dicts(counting_dicterizer) == [{'Reference_A': 1}, {'Reference_B': 1}]


def test_timeline_to_sklearn_dataset():
    timeline: Timeline = [Reference('Reference_A', now), Reference('Reference_A', now), Reference('Reference_B', now)]
    timeline_dataset = TimelineDataset([timeline], [FeatureClass.POSITIVE], [False])
    sklearn_dataset = timeline_to_sklearn_dataset(timeline_dataset, counting_dicterizer)
    assert np.all(sklearn_dataset.X.toarray() == [2, 1])
    assert sklearn_dataset.y == [1]
    assert len(sklearn_dataset.splits) == 1
    train_split = sklearn_dataset.splits[0][0]
    test_split = sklearn_dataset.splits[0][1]
    assert len(train_split) + len(test_split) == 1
    assert not any(map(lambda x: x in train_split, test_split))
    assert not any(map(lambda x: x in test_split, train_split))


def test_timeline_dataset_test_indices():
    timeline: Timeline = [Reference('Reference_A', now)]
    dataset = TimelineDataset([timeline, timeline, timeline],
                              [FeatureClass.POSITIVE, FeatureClass.POSITIVE, FeatureClass.NEGATIVE],
                              [False, True, True])
    assert dataset.test_indices() == [1, 2]


def test_metrics():
    timeline: Timeline = [Reference('Reference_A', now)]
    dataset = TimelineDataset([timeline, timeline, timeline],
                              [FeatureClass.POSITIVE, FeatureClass.POSITIVE, FeatureClass.NEGATIVE],
                              [False, False, True])
    metrics = dataset.metrics()
    assert metrics.training_datasets == 2
    assert metrics.test_datasets == 1
    assert metrics.test_datasets_in_all == 1/3
    assert metrics.positive_classes == 2
    assert metrics.negative_classes == 1
    assert metrics.positive_classes_in_all == 2/3
