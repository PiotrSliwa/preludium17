from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Optional

EntityName = str


@dataclass(frozen=True)
class Reference:
    name: EntityName
    date: datetime


Timeline = List[Reference]
DateSpan = Tuple[datetime, datetime]


def timeline_date_span(timeline: Timeline) -> DateSpan:
    return timeline[0].date, timeline[-1].date


def timeline_filter_out(timeline: Timeline, entity_name: EntityName) -> Timeline:
    return list(filter(lambda reference: reference.name != entity_name, timeline))


def __splitting_index(timeline: Timeline, timepoint: datetime) -> Optional[int]:
    for index, reference in enumerate(timeline):
        if reference.date >= timepoint:
            return index
    return None


def timeline_split_by_timepoint(timeline: Timeline, timepoint: datetime) -> Tuple[Timeline, Timeline]:
    index = __splitting_index(timeline, timepoint)
    return timeline[:index], timeline[index:]