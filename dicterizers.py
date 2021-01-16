from timelines import Timeline
from datasets import FeatureDict


def counting_dicterizer(timeline: Timeline) -> FeatureDict:
    result: FeatureDict = {}
    for reference in timeline:
        current = result.setdefault(reference.name, 0)
        result[reference.name] = current + 1
    return result

