from datetime import datetime

from timelines import timeline_date_span, Reference, Timeline, timeline_filter_out

now = datetime.now()


def test_timeline_date_span():
    first = datetime(2000, 1, 1)
    second = datetime(2001, 1, 2)
    third = datetime(2001, 1, 3)
    span = timeline_date_span([Reference(name='Reference_A', date=first),
                               Reference(name='Reference_B', date=second),
                               Reference(name='Reference_C', date=third)])
    assert span == (first, third)


def test_timeline_date_span_mixed():
    first = datetime(2000, 1, 1)
    second = datetime(2001, 1, 2)
    third = datetime(2001, 1, 3)
    span = timeline_date_span([Reference(name='Reference_A', date=first),
                               Reference(name='Reference_C', date=third),
                               Reference(name='Reference_B', date=second)])
    assert span == (first, second)


def test_timeline_filter_out():
    timeline: Timeline = [Reference(name='Reference_A', date=now),
                          Reference(name='Reference_B', date=now)]
    assert timeline_filter_out(timeline, 'Reference_A') == [Reference(name='Reference_B', date=now)]
