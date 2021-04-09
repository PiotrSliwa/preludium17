import collections
from dataclasses import dataclass
from datetime import datetime, timedelta
from random import random
from typing import Callable, List, Optional, Dict, Deque

from focals import Focal
from timelines import Timeline, EntityName, timeline_filter_out, timeline_split_by_timepoint, timeline_date_span, Reference
from datasets import TimelineDataset, FeatureClass
from lists import last_index, indexes_of


class TimelineProcessor:
    def __call__(self, timeline: Timeline) -> TimelineDataset: ...


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


@dataclass
class SlicingProcessor(TimelineProcessor):
    entity_name: EntityName
    timepoint: datetime

    def __call__(self, timeline: Timeline) -> TimelineDataset:
        indexes = indexes_of(timeline, lambda reference: reference.name == self.entity_name)
        last = 0
        x: List[Timeline] = []
        y: List[FeatureClass] = []
        test: List[bool] = []
        for current in indexes:
            x.append(timeline[last:current])
            y.append(FeatureClass.POSITIVE)
            test.append(timeline[current].date >= self.timepoint)
            last = current + 1
        if last < len(timeline):
            x.append(timeline[last:])
            y.append(FeatureClass.NEGATIVE)
            test.append(timeline[last].date >= self.timepoint)
        return TimelineDataset(x, y, test)


@dataclass
class WindowingProcessor(TimelineProcessor):
    entity_name: EntityName
    timepoint: datetime
    limit: timedelta

    def __call__(self, timeline: Timeline) -> TimelineDataset:
        break_index = last_index(timeline, lambda reference: reference.name == self.entity_name)
        unbounded = self._unbounded_window(timeline[break_index:]) if break_index is not None else self._unbounded_window(timeline)
        if break_index is None:
            return unbounded
        else:
            return self._bounded_window(timeline[0:break_index]) + unbounded

    def _bounded_window(self, timeline: Timeline) -> TimelineDataset:
        result = TimelineDataset()
        bucket: Deque[Reference] = collections.deque()
        feature_class: FeatureClass = FeatureClass.POSITIVE
        next_turnover: Optional[int] = None
        stack = list(range(len(timeline)))
        while stack:
            i = stack.pop()
            reference = timeline[i]
            if reference.name == self.entity_name:
                if feature_class == FeatureClass.NEGATIVE:
                    feature_class = FeatureClass.POSITIVE
                    bucket.clear()
                    continue
                elif next_turnover is None:
                    next_turnover = i
            else:
                if len(bucket) > 0 and bucket[-1].date - reference.date >= self.limit:
                    result += TimelineDataset([list(bucket)], [feature_class], [self.timepoint < bucket[-1].date])
                    bucket.clear()
                    feature_class = FeatureClass.NEGATIVE
                    if next_turnover is not None:
                        stack = list(range(next_turnover + 1))
                        next_turnover = None
            bucket.appendleft(reference)
        return result

    def _unbounded_window(self, timeline: Timeline) -> TimelineDataset:
        result = TimelineDataset()
        bucket: Timeline = []
        for reference in timeline:
            if len(bucket) > 0 and reference.date - bucket[0].date >= self.limit:
                result += TimelineDataset([list(bucket)], [FeatureClass.NEGATIVE], [self.timepoint < reference.date])
                bucket.clear()
            bucket.append(reference)
        return result


def focals_to_timeline_dataset(focals: List[Focal], processor: TimelineProcessor) -> TimelineDataset:
    result = TimelineDataset()
    for focal in focals:
        dataset = processor(focal.timeline)
        result += dataset
    return result
