from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, Tuple, Dict, List

from timelines import EntityName, Timeline, timeline_date_span, DateSpan


@dataclass(frozen=True)
class Focal:
    name: EntityName
    timeline: Timeline


class FocalGroupSpan:

    @dataclass(frozen=True)
    class Boundary:
        timepoint: datetime
        focals: List[EntityName]

    boundaries: List[Boundary]

    def __init__(self, focals: Iterator[Focal]):
        boundaries: Dict[datetime, List[EntityName]] = {}
        for focal in focals:
            span = timeline_date_span(focal.timeline)
            boundaries.setdefault(span[0], []).append(focal.name)
            boundaries.setdefault(span[1], []).append(focal.name)
        self.boundaries = [FocalGroupSpan.Boundary(e[0], e[1]) for e in sorted(boundaries.items())]

    def outer(self) -> DateSpan:
        return self.boundaries[0].timepoint, self.boundaries[-1].timepoint


