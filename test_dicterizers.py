from datetime import datetime

from database import Timeline, Reference
from dicterizers import counting_dicterizer


now = datetime.now()


def test_counting_dicterizer():
    timeline: Timeline = [
        Reference('A', now),
        Reference('B', now),
        Reference('A', now)
    ]
    assert counting_dicterizer(timeline) == {'A': 2, 'B': 1}
