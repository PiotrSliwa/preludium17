import csv
from dataclasses import dataclass
from datetime import datetime
from typing import List

from database import Database
from focals import FocalGroupSpan, Focal, DistributionPoint


@dataclass
class PointStats:
    lower: int
    higher: int
    first: datetime
    last: datetime


def point_stats(focals: List[Focal], highest_distribution_point: DistributionPoint) -> PointStats:
    dates = [reference.date for focal in focals for reference in focal.timeline]
    lower = [date < highest_distribution_point.timepoint for date in dates]
    return PointStats(lower=lower.count(True), higher=lower.count(False), first=min(dates), last=max(dates))


if __name__ == '__main__':
    database = Database()
    focals = database.get_focals()
    focal_group_span = FocalGroupSpan(focals)
    highest_distribution_point = focal_group_span.highest_distribution_points()[0]
    print(f'Highest distribution point: {highest_distribution_point}')
    stats = point_stats(focals, highest_distribution_point)
    print(f'Stats: {stats}')
    with open('distribution.csv', 'w') as f:
        writer = csv.DictWriter(f, dialect='excel', fieldnames=['timepoint', 'focals'])
        for point in focal_group_span.distribution():
            writer.writerow(point.__dict__)
    print('Done.')
