from focals import Focal, FocalGroupSpan
from test_utils import day
from timelines import Reference


def test_focal_group_span():
    focal_a = Focal('Focal_A', [Reference('Reference_C', day[2]), Reference('Reference_D', day[3])])
    focal_b = Focal('Focal_B', [Reference('Reference_A', day[1]), Reference('Reference_B', day[2])])
    focal_c = Focal('Focal_C', [Reference('Reference_E', day[2]),
                                Reference('Reference_F', day[3]),
                                Reference('Reference_G', day[4])])
    focal_d = Focal('Focal_D', [Reference('Reference_H', day[3])])
    focals = [focal_a, focal_b, focal_c, focal_d]
    span = FocalGroupSpan(focals)
    assert span.points == [FocalGroupSpan.Point(day[1], {focal_b.name}),
                           FocalGroupSpan.Point(day[2], {focal_a.name, focal_b.name, focal_c.name}),
                           FocalGroupSpan.Point(day[3], {focal_a.name, focal_c.name, focal_d.name}),
                           FocalGroupSpan.Point(day[4], {focal_c.name})]
    assert span.distribution() == [FocalGroupSpan.DistributionPoint(day[1], 1),
                                   FocalGroupSpan.DistributionPoint(day[2], 3),
                                   FocalGroupSpan.DistributionPoint(day[3], 3),
                                   FocalGroupSpan.DistributionPoint(day[4], 1)]
    assert span.highest_distribution_points() == [FocalGroupSpan.DistributionPoint(day[2], 3),
                                                  FocalGroupSpan.DistributionPoint(day[3], 3)]
    assert span.outer() == (day[1], day[4])
