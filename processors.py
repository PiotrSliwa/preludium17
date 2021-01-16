from dataclasses import dataclass
from random import random
from typing import Callable, List

from database import Focal
from timelines import Timeline, EntityName
from datasets import TimelineDataset, FeatureClass
from lists import last_index

TimelineProcessor = Callable[[Timeline], TimelineDataset]


@dataclass
class FilterAndSliceToMostRecentProcessor(TimelineProcessor):
    entity_name: EntityName

    def __flip_coin(self, probability=.2) -> bool:
        if random() <= probability:
            return True
        return False

    def __call__(self, timeline: Timeline) -> TimelineDataset:
        index = last_index(timeline, lambda reference: reference.name == self.entity_name)
        if index is None:
            return TimelineDataset([timeline], [FeatureClass.NEGATIVE], [self.__flip_coin()])
        sub_timeline = timeline[:index]
        filtered_sub_timeline = list(filter(lambda reference: reference.name != self.entity_name, sub_timeline))
        return TimelineDataset([filtered_sub_timeline], [FeatureClass.POSITIVE], [self.__flip_coin()])

    def __str__(self):
        return f'FilterAndSliceToMostRecentProcessor(entity_name={self.entity_name})'


def focals_to_timeline_dataset(focals: List[Focal], processor: TimelineProcessor) -> TimelineDataset:
    result = TimelineDataset()
    for focal in focals:
        dataset = processor(focal.timeline)
        result += dataset
    return result
