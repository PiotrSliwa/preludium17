from random import random
from typing import Callable, List

from database import Timeline, EntityName, Focal
from datasets import TimelineDataset, FeatureClass
from lists import last_index

TimelineProcessor = Callable[[Timeline, EntityName], TimelineDataset]


def __flip_coin(probability=.2) -> bool:
    if random() <= probability:
        return True
    return False


def filter_and_slice_to_most_recent(timeline: Timeline, entity_name: EntityName) -> TimelineDataset:
    index = last_index(timeline, lambda reference: reference.name == entity_name)
    if index is None:
        return TimelineDataset([timeline], [FeatureClass.NEGATIVE], [__flip_coin()])
    sub_timeline = timeline[:index]
    filtered_sub_timeline = list(filter(lambda reference: reference.name != entity_name, sub_timeline))
    return TimelineDataset([filtered_sub_timeline], [FeatureClass.POSITIVE], [__flip_coin()])


def focals_to_timeline_dataset(focals: List[Focal], entity_name: EntityName,
                               processor: TimelineProcessor) -> TimelineDataset:
    result = TimelineDataset()
    for focal in focals:
        dataset = processor(focal.timeline, entity_name)
        result += dataset
    return result
