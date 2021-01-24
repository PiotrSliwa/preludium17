from test_utils import now, day_first, day_second, day_third, day_fourth
from timelines import Reference
from dicterizers import counting_dicterizer
from processors import *


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
