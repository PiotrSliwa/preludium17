from typing import Callable, List

from database import Timeline, EntityName, Focal
from datasets import TimelineDataset, FeatureClass
from lists import last_index

Preprocessor = Callable[[Timeline], TimelineDataset]


class EntityPreprocessor:
    def __init__(self, entity_name: EntityName):
        self.__entity_name = entity_name

    def filter_and_slice_to_most_recent(self, timeline: Timeline) -> TimelineDataset:
        index = last_index(timeline, lambda reference: reference.name == self.__entity_name)
        if index is None:
            return TimelineDataset([timeline], [FeatureClass.NEGATIVE])
        sub_timeline = timeline[:index]
        filtered_sub_timeline = list(filter(lambda reference: reference.name != self.__entity_name, sub_timeline))
        return TimelineDataset([filtered_sub_timeline], [FeatureClass.POSITIVE])


def focals_to_timeline_dataset(focals: List[Focal], preprocessor: Preprocessor) -> TimelineDataset:
    result = TimelineDataset()
    for focal in focals:
        dataset = preprocessor(focal.timeline)
        result += dataset
    return result

