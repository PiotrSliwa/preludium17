from test_utils import now, day
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
    timeline: Timeline = [Reference(name='Reference_A', date=day[1]),
                          Reference(name='Reference_B', date=day[2]),
                          Reference(name='Reference_C', date=day[3]),
                          Reference(name='Reference_D', date=day[4])]
    processor = TimepointProcessor(entity_name, day[3])
    result = processor(timeline)
    assert result.feature_dicts(counting_dicterizer) == [{'Reference_A': 1},
                                                         {'Reference_C': 1, 'Reference_D': 1}]
    assert result.feature_classes() == [FeatureClass.POSITIVE, FeatureClass.NEGATIVE]
    assert result.test_indices() == [1]


def test_slicing_processor_containing_entity():
    entity_name = 'Reference_X'
    timeline: Timeline = [Reference(name='Reference_1', date=day[1]),
                          Reference(name='Reference_2', date=day[2]),
                          Reference(name=entity_name, date=day[3]),
                          Reference(name='Reference_4', date=day[4]),
                          Reference(name='Reference_5', date=day[5]),
                          Reference(name='Reference_6', date=day[6]),
                          Reference(name='Reference_7', date=day[7]),
                          Reference(name=entity_name, date=day[8]),
                          Reference(name='Reference_9', date=day[9])]
    processor = SlicingProcessor(entity_name, day[5])
    result = processor(timeline)
    assert result.feature_dicts(counting_dicterizer) == [{'Reference_1': 1, 'Reference_2': 1},
                                                         {'Reference_4': 1, 'Reference_5': 1, 'Reference_6': 1, 'Reference_7': 1},
                                                         {'Reference_9': 1}]
    assert result.feature_classes() == [FeatureClass.POSITIVE, FeatureClass.POSITIVE, FeatureClass.NEGATIVE]
    assert result.test_indices() == [1, 2]


def test_slicing_processor_not_containing_entity():
    entity_name = 'Reference_X'
    timeline: Timeline = [Reference(name='Reference_1', date=day[1]),
                          Reference(name='Reference_2', date=day[2])]
    processor = SlicingProcessor(entity_name, day[2])
    result = processor(timeline)
    assert result.feature_dicts(counting_dicterizer) == [{'Reference_1': 1, 'Reference_2': 1}]
    assert result.feature_classes() == [FeatureClass.NEGATIVE]
    assert result.test_indices() == []


def test_windowing_processor_containing_entities_sparsely():
    entity_name = 'Reference_X'
    timeline: Timeline = [Reference(name='Reference_1', date=day[1]),
                          Reference(name='Reference_2', date=day[2]),
                          Reference(name='Reference_3', date=day[3]),
                          Reference(name=entity_name, date=day[4]),
                          Reference(name='Reference_5', date=day[5]),
                          Reference(name='Reference_6', date=day[6]),
                          Reference(name='Reference_7', date=day[7]),
                          Reference(name='Reference_8', date=day[8]),
                          Reference(name='Reference_9', date=day[9]),
                          Reference(name=entity_name, date=day[10]),
                          Reference(name='Reference_11', date=day[11])]
    processor = WindowingProcessor(entity_name, day[6], timedelta(days=2))
    result = processor(timeline)
    assert result.feature_dicts(counting_dicterizer) == [{'Reference_8': 1, 'Reference_9': 1},
                                                         {'Reference_6': 1, 'Reference_7': 1},
                                                         {'Reference_2': 1, 'Reference_3': 1},]
    assert result.feature_classes() == [FeatureClass.POSITIVE,
                                        FeatureClass.NEGATIVE,
                                        FeatureClass.POSITIVE]
    assert result.test_indices() == [0, 1]


def test_windowing_processor_containing_entities_densely():
    entity_name = 'Reference_X'
    timeline: Timeline = [Reference(name='Reference_1', date=day[1]),
                          Reference(name='Reference_2', date=day[2]),
                          Reference(name=entity_name, date=day[3]),
                          Reference(name='Reference_4', date=day[4]),
                          Reference(name=entity_name, date=day[5]),
                          Reference(name='Reference_6', date=day[6])]
    processor = WindowingProcessor(entity_name, day[3], timedelta(days=3))
    result = processor(timeline)
    assert result.feature_dicts(counting_dicterizer) == [{entity_name: 1, 'Reference_2': 1, 'Reference_4': 1}]
    assert result.feature_classes() == [FeatureClass.POSITIVE]
    assert result.test_indices() == [0]


def test_windowing_processor_with_limit_not_containing_entity():
    entity_name = 'Reference_X'
    timeline: Timeline = [Reference(name='Reference_1', date=day[1]),
                          Reference(name='Reference_2', date=day[2]),
                          Reference(name='Reference_3', date=day[3])]
    processor = WindowingProcessor(entity_name, day[2], timedelta(days=1))
    result = processor(timeline)
    assert result.feature_dicts(counting_dicterizer) == [{'Reference_1': 1}, {'Reference_2': 1}]
    assert result.feature_classes() == [FeatureClass.NEGATIVE, FeatureClass.NEGATIVE]
    assert result.test_indices() == [1]


def test_windowing_processor_with_limit_not_containing_entity_2_day():
    entity_name = 'Reference_X'
    timeline: Timeline = [Reference(name='Reference_1', date=day[1]),
                          Reference(name='Reference_2', date=day[2]),
                          Reference(name='Reference_3', date=day[3])]
    processor = WindowingProcessor(entity_name, day[3], timedelta(days=2))
    result = processor(timeline)
    assert result.feature_dicts(counting_dicterizer) == [{'Reference_1': 1, 'Reference_2': 1}]
    assert result.feature_classes() == [FeatureClass.NEGATIVE]
    assert result.test_indices() == []
