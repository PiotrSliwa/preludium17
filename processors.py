from dataclasses import dataclass
from datetime import datetime
from random import random
from typing import Callable, List, Optional, Dict

from focals import Focal
from timelines import Timeline, EntityName, timeline_filter_out, timeline_split_by_timepoint
from datasets import TimelineDataset, FeatureClass
from lists import last_index


class TimelineProcessor:
    def __call__(self, timeline: Timeline) -> TimelineDataset: ...
    def to_dict(self) -> Dict: ...


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

    def to_dict(self) -> Dict:
        return {'type': 'FilterAndSliceToMostRecentProcessor', 'entity_name': self.entity_name}


@dataclass
class TimepointProcessor(TimelineProcessor):
    entity_name: EntityName
    timepoint: datetime

    def __feature_class(self, timeline: Timeline) -> FeatureClass:
        contains = any(self.entity_name == r.name for r in timeline)
        return FeatureClass.POSITIVE if contains else FeatureClass.NEGATIVE

    def __call__(self, timeline: Timeline) -> TimelineDataset:
        training_timeline, test_timeline = timeline_split_by_timepoint(timeline, self.timepoint)
        training_class = self.__feature_class(training_timeline)
        test_class = self.__feature_class(test_timeline)
        x = [timeline_filter_out(training_timeline, self.entity_name),
             timeline_filter_out(test_timeline, self.entity_name)]
        y = [training_class, test_class]
        test = [False, True]
        return TimelineDataset(x, y, test)

    def to_dict(self) -> Dict:
        return {'type': 'TimepointProcessor', 'entity_name': self.entity_name, 'timepoint': self.timepoint}


def focals_to_timeline_dataset(focals: List[Focal], processor: TimelineProcessor) -> TimelineDataset:
    result = TimelineDataset()
    for focal in focals:
        dataset = processor(focal.timeline)
        result += dataset
    return result
