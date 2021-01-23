from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

EntityName = str


@dataclass(frozen=True)
class Reference:
    name: EntityName
    date: datetime


Timeline = List[Reference]


def timeline_date_span(timeline: Timeline) -> Tuple[datetime, datetime]:
    return timeline[0].date, timeline[-1].date


def timeline_filter_out(timeline: Timeline, entity_name: EntityName) -> Timeline:
    return list(filter(lambda reference: reference.name != entity_name, timeline))
