from datetime import datetime

from database import Timeline, Reference
from datasets import FeatureClass
from dicterizers import counting_dicterizer
from processors import EntityPreprocessor

now = datetime.now()


def test_filter_and_slice_to_most_recent_having_the_reference():
    filtered_entity = 'Reference_B'
    preprocessor = EntityPreprocessor(filtered_entity)
    timeline: Timeline = [Reference(name='Reference_A', date=now),
                          Reference(name='Reference_B', date=now),
                          Reference(name='Reference_A', date=now),
                          Reference(name='Reference_B', date=now),
                          Reference(name='Reference_C', date=now)]
    result = preprocessor.filter_and_slice_to_most_recent(timeline)
    assert result.feature_dicts(counting_dicterizer) == [{'Reference_A': 2}]
    assert result.feature_classes() == [FeatureClass.POSITIVE]


def test_filter_and_slice_to_most_recent_but_without_the_reference():
    filtered_entity = 'SomethingDifferent'
    preprocessor = EntityPreprocessor(filtered_entity)
    timeline: Timeline = [Reference(name='Reference_A', date=now)]
    result = preprocessor.filter_and_slice_to_most_recent(timeline)
    assert result.feature_dicts(counting_dicterizer) == [{'Reference_A': 1}]
    assert result.feature_classes() == [FeatureClass.NEGATIVE]