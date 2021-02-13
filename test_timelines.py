from test_utils import *
from timelines import timeline_date_span, Reference, Timeline, timeline_filter_out, timeline_split_by_timepoint


def test_timeline_date_span():
    span = timeline_date_span([Reference(name='Reference_A', date=day_first),
                               Reference(name='Reference_B', date=day_second),
                               Reference(name='Reference_C', date=day_third)])
    assert span == (day_first, day_third)


def test_timeline_date_span_mixed():
    span = timeline_date_span([Reference(name='Reference_A', date=day_first),
                               Reference(name='Reference_C', date=day_third),
                               Reference(name='Reference_B', date=day_second)])
    assert span == (day_first, day_second)


def test_timeline_filter_out():
    timeline: Timeline = [Reference(name='Reference_A', date=now),
                          Reference(name='Reference_B', date=now)]
    assert timeline_filter_out(timeline, 'Reference_A') == [Reference(name='Reference_B', date=now)]


def test_timeline_split_by_timepoint():
    timeline: Timeline = [Reference(name='Reference_A', date=day_first),
                          Reference(name='Reference_B', date=day_second),
                          Reference(name='Reference_C', date=day_third)]
    split = timeline_split_by_timepoint(timeline, day_second)
    assert split[0] == [Reference(name='Reference_A', date=day_first)]
    assert split[1] == [Reference(name='Reference_B', date=day_second),
                        Reference(name='Reference_C', date=day_third)]
