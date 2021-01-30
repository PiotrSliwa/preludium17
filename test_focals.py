from focals import Focal, FocalGroupSpan
from test_utils import day_first, day_second, day_third, day_fourth
from timelines import Reference


def test_focal_group_span():
    focal_a = Focal('Focal_A', [Reference('Reference_C', day_second), Reference('Reference_D', day_third)])
    focal_b = Focal('Focal_B', [Reference('Reference_A', day_first), Reference('Reference_B', day_second)])
    focal_c = Focal('Focal_C', [Reference('Reference_E', day_second),
                                Reference('Reference_F', day_third),
                                Reference('Reference_G', day_fourth)])
    focal_d = Focal('Focal_D', [Reference('Reference_H', day_third)])
    focals = [focal_a, focal_b, focal_c, focal_d]
    span = FocalGroupSpan(focals)
    assert span.points == [FocalGroupSpan.Point(day_first, {focal_b.name}),
                           FocalGroupSpan.Point(day_second, {focal_a.name, focal_b.name, focal_c.name}),
                           FocalGroupSpan.Point(day_third, {focal_a.name, focal_c.name, focal_d.name}),
                           FocalGroupSpan.Point(day_fourth, {focal_c.name})]
    assert span.distribution() == [FocalGroupSpan.DistributionPoint(day_first, 1),
                                   FocalGroupSpan.DistributionPoint(day_second, 3),
                                   FocalGroupSpan.DistributionPoint(day_third, 3),
                                   FocalGroupSpan.DistributionPoint(day_fourth, 1)]
    assert span.highest_distribution_points() == [FocalGroupSpan.DistributionPoint(day_second, 3),
                                                  FocalGroupSpan.DistributionPoint(day_third, 3)]
    assert span.outer() == (day_first, day_fourth)
