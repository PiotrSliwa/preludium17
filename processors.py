from dataclasses import dataclass
from datetime import datetime
from random import random
from typing import Callable, List, Optional

from focals import Focal
from timelines import Timeline, EntityName, timeline_filter_out
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
        filtered_sub_timeline = timeline_filter_out(sub_timeline, self.entity_name)
        return TimelineDataset([filtered_sub_timeline], [FeatureClass.POSITIVE], [self.__flip_coin()])

    def __str__(self):
        return f'FilterAndSliceToMostRecentProcessor(entity_name={self.entity_name})'


@dataclass
class TimepointProcessor(TimelineProcessor):
    entity_name: EntityName
    timepoint: datetime

    def __splitting_index(self, timeline: Timeline) -> Optional[int]:
        for index, reference in enumerate(timeline):
            if reference.date >= self.timepoint:
                return index
        return None

    def __feature_class(self, timeline: Timeline) -> FeatureClass:
        contains = any(self.entity_name == r.name for r in timeline)
        return FeatureClass.POSITIVE if contains else FeatureClass.NEGATIVE

    def __call__(self, timeline: Timeline) -> TimelineDataset:
        index = self.__splitting_index(timeline)
        training_timeline = timeline[:index]
        training_class = self.__feature_class(training_timeline)
        test_timeline = timeline[index:]
        test_class = self.__feature_class(test_timeline)
        x = [timeline_filter_out(training_timeline, self.entity_name),
             timeline_filter_out(test_timeline, self.entity_name)]
        y = [training_class, test_class]
        test = [False, True]
        return TimelineDataset(x, y, test)

    def __str__(self):
        return f'TimepointProcessor(entity_name={self.entity_name},timepoint={self.timepoint})'


def focals_to_timeline_dataset(focals: List[Focal], processor: TimelineProcessor) -> TimelineDataset:
    result = TimelineDataset()
    for focal in focals:
        dataset = processor(focal.timeline)
        result += dataset
    return result
