from dataclasses import dataclass
import time
from datetime import datetime
from typing import Iterator, Tuple, Dict, List, Optional, Set

from timelines import EntityName, Timeline, timeline_date_span, DateSpan


@dataclass(frozen=True)
class Focal:
    name: EntityName
    timeline: Timeline


@dataclass
class FocalGroupSpan:

    @dataclass(frozen=True)
    class Point:
        timepoint: datetime
        focals: Set[EntityName]

    @dataclass(frozen=True)
    class DistributionPoint:
        timepoint: datetime
        focals: int

    points: List[Point]

    def __init__(self, focals: Iterator[Focal]):
        spans: Dict[EntityName, DateSpan] = {focal.name: timeline_date_span(focal.timeline) for focal in focals}
        time_points: Set[datetime] = set()
        for span in spans.values():
            time_points.add(span[0])
            time_points.add(span[1])
        self.points = []
        for time_point in time_points:
            timepoint_focals: Set[EntityName] = set()
            for focal_name, span in spans.items():
                if span[0] <= time_point <= span[1]:
                    timepoint_focals.add(focal_name)
            self.points.append(FocalGroupSpan.Point(time_point, timepoint_focals))
        self.points.sort(key=lambda p: p.timepoint)

    def outer(self) -> DateSpan:
        return self.points[0].timepoint, self.points[-1].timepoint

    def distribution(self) -> List[DistributionPoint]:
        return [FocalGroupSpan.DistributionPoint(point.timepoint, len(point.focals)) for point in self.points]

