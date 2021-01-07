import itertools
from pprint import pprint
from typing import List, NamedTuple

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from database import Database, Focal
from datasets import TimelineDataset, timeline_to_sklearn_dataset, Dicterizer
from dicterizers import counting_dicterizer
from preprocessors import Preprocessor, EntityPreprocessor


def focals_to_timeline_dataset(focals: List[Focal], preprocessor: Preprocessor) -> TimelineDataset:
    result = TimelineDataset()
    for focal in focals:
        dataset = preprocessor(focal.timeline)
        result += dataset
    return result


class BenchmarkResult(NamedTuple):
    preprocessor: str
    dicterizer: str
    clf: str
    scores: List[float]


def benchmark(focals: List[Focal], preprocessors: List[Preprocessor], dicterizers: List[Dicterizer], clfs: List) -> List[BenchmarkResult]:
    results: List[BenchmarkResult] = []
    inputs = list(itertools.product(preprocessors, dicterizers, clfs))
    i = 1
    for preprocessor, dicterizer, clf in inputs:
        timeline_dataset = focals_to_timeline_dataset(focals, preprocessor)
        sklearn_dataset = timeline_to_sklearn_dataset(timeline_dataset, counting_dicterizer)
        clf = DecisionTreeClassifier()
        scores = cross_val_score(clf, sklearn_dataset.X, sklearn_dataset.y, cv=3)
        results.append(BenchmarkResult(preprocessor=preprocessor.__name__, dicterizer=dicterizer.__name__, clf=str(clf),
                                       scores=scores))
        print(f'Benchmark iteration: {i} / {len(inputs)}')
        i += 1
    return results


def main():
    database = Database()
    focals = database.get_focals()
    references = list(database.get_averagely_popular_references(precision=1))
    results = {}
    i = 1
    for reference in references:
        entity_name = reference.name
        entity_preprocessor = EntityPreprocessor(entity_name)
        print(f'Entity {entity_name} ({i} / {len(references)})')
        result = benchmark(focals,
                           preprocessors=[entity_preprocessor.filter_and_slice_to_most_recent],
                           dicterizers=[counting_dicterizer],
                           clfs=[DecisionTreeClassifier()])
        results[entity_name] = result
        print('---')
        i += 1
    pprint(results)


if __name__ == '__main__':
    main()
