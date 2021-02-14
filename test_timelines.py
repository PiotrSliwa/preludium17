from test_utils import *
from timelines import timeline_date_span, Reference, Timeline, timeline_filter_out, timeline_split_by_timepoint


def test_timeline_date_span():
    span = timeline_date_span([Reference(name='Reference_A', date=day[1]),
                               Reference(name='Reference_B', date=day[2]),
                               Reference(name='Reference_C', date=day[3])])
    assert span == (day[1], day[3])


def test_timeline_date_span_mixed():
    span = timeline_date_span([Reference(name='Reference_A', date=day[1]),
                               Reference(name='Reference_C', date=day[3]),
                               Reference(name='Reference_B', date=day[2])])
    assert span == (day[1], day[2])


def test_timeline_filter_out():
    timeline: Timeline = [Reference(name='Reference_A', date=now),
                          Reference(name='Reference_B', date=now)]
    assert timeline_filter_out(timeline, 'Reference_A') == [Reference(name='Reference_B', date=now)]


def test_timeline_split_by_timepoint():
    timeline: Timeline = [Reference(name='Reference_A', date=day[1]),
                          Reference(name='Reference_B', date=day[2]),
                          Reference(name='Reference_C', date=day[3])]
    split = timeline_split_by_timepoint(timeline, day[2])
    assert split[0] == [Reference(name='Reference_A', date=day[1])]
    assert split[1] == [Reference(name='Reference_B', date=day[2]),
                        Reference(name='Reference_C', date=day[3])]
