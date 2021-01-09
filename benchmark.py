import itertools
from dataclasses import dataclass
from pprint import pprint
from typing import List

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from database import Database, Focal, EntityName
from datasets import timeline_to_sklearn_dataset, Dicterizer
from dicterizers import counting_dicterizer
from preprocessors import Preprocessor, EntityPreprocessor, focals_to_timeline_dataset


@dataclass(frozen=True)
class BenchmarkResult:
    preprocessor: str
    dicterizer: str
    clf: str
    scores: List[float]


@dataclass(frozen=True)
class EntityBenchmarkResult(BenchmarkResult):
    entity_name: EntityName

    @staticmethod
    def from_benchmark_result(result: BenchmarkResult, entity_name: EntityName):
        return EntityBenchmarkResult(preprocessor=result.preprocessor, dicterizer=result.dicterizer, clf=result.clf,
                                     scores=result.scores, entity_name=entity_name)


def benchmark(focals: List[Focal], preprocessors: List[Preprocessor], dicterizers: List[Dicterizer], clfs: List) -> \
List[BenchmarkResult]:
    results: List[BenchmarkResult] = []
    i = 1
    for preprocessor in preprocessors:
        inputs = list(itertools.product(dicterizers, clfs))
        for dicterizer, clf in inputs:
            timeline_dataset = focals_to_timeline_dataset(focals, preprocessor)
            sklearn_dataset = timeline_to_sklearn_dataset(timeline_dataset, counting_dicterizer)
            clf = DecisionTreeClassifier()
            scores = cross_val_score(clf, sklearn_dataset.X, sklearn_dataset.y, cv=3)
            results.append(BenchmarkResult(preprocessor=preprocessor.__name__,
                                           dicterizer=dicterizer.__name__,
                                           clf=str(clf),
                                           scores=scores))
            print(f'Benchmark iteration: {i} / {len(inputs) + len(preprocessors)}')
            i += 1
    return results


def main():
    database = Database()
    focals = database.get_focals()
    references = list(database.get_averagely_popular_references(precision=1))
    results: List[EntityBenchmarkResult] = []
    i = 1
    for reference in references:
        entity_name = reference.name
        entity_preprocessor = EntityPreprocessor(entity_name)
        print(f'Entity {entity_name} ({i} / {len(references)})')
        result = benchmark(focals,
                           preprocessors=[entity_preprocessor.filter_and_slice_to_most_recent],
                           dicterizers=[counting_dicterizer],
                           clfs=[DecisionTreeClassifier()])
        results += list(map(lambda r: EntityBenchmarkResult.from_benchmark_result(r, entity_name), result))
        i += 1
    pprint(results)


if __name__ == '__main__':
    main()
