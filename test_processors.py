from datetime import datetime

import pytest

from timelines import Timeline, Reference
from datasets import FeatureClass
from dicterizers import counting_dicterizer
from processors import *

now = datetime.now()
day_first = datetime(2000, 1, 1)
day_second = datetime(2000, 1, 2)
day_third = datetime(2000, 1, 3)
day_fourth = datetime(2000, 1, 4)


def test_filter_and_slice_to_most_recent_having_the_reference():
    filtered_entity = 'Reference_B'
    timeline: Timeline = [Reference(name='Reference_A', date=now),
                          Reference(name='Reference_B', date=now),
                          Reference(name='Reference_A', date=now),
                          Reference(name='Reference_B', date=now),
                          Reference(name='Reference_C', date=now)]
    processor = FilterAndSliceToMostRecentProcessor(filtered_entity)
    result = processor(timeline)
    assert result.feature_dicts(counting_dicterizer) == [{'Reference_A': 2}]
    assert result.feature_classes() == [FeatureClass.POSITIVE]


def test_filter_and_slice_to_most_recent_but_without_the_reference():
    filtered_entity = 'SomethingDifferent'
    timeline: Timeline = [Reference(name='Reference_A', date=now)]
    processor = FilterAndSliceToMostRecentProcessor(filtered_entity)
    result = processor(timeline)
    assert result.feature_dicts(counting_dicterizer) == [{'Reference_A': 1}]
    assert result.feature_classes() == [FeatureClass.NEGATIVE]


def test_timepoint():
    entity_name = 'Reference_B'
    timeline: Timeline = [Reference(name='Reference_A', date=day_first),
                          Reference(name='Reference_B', date=day_second),
                          Reference(name='Reference_C', date=day_third),
                          Reference(name='Reference_D', date=day_fourth)]
    processor = TimepointProcessor(entity_name, day_third)
    result = processor(timeline)
    assert result.feature_dicts(counting_dicterizer) == [{'Reference_A': 1},
                                                         {'Reference_C': 1, 'Reference_D': 1}]
    assert result.feature_classes() == [FeatureClass.POSITIVE, FeatureClass.NEGATIVE]
    assert result.test_indices() == [1]
