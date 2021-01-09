import itertools
from dataclasses import dataclass
from pprint import pprint
from typing import List

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from database import Database, Focal, EntityName
from datasets import timeline_to_sklearn_dataset, Dicterizer
from dicterizers import counting_dicterizer
from processors import focals_to_timeline_dataset, filter_and_slice_to_most_recent, TimelineProcessor
import statistics


@dataclass(frozen=True)
class EntityBenchmarkResult:
    entity_name: EntityName
    preprocessor: str
    dicterizer: str
    clf: str
    scores: List[float]
    score_avg: float
    score_std: float


def benchmark(focals: List[Focal], entity_names: List[EntityName], processors: List[TimelineProcessor],
              dicterizers: List[Dicterizer], clfs: List) -> List[EntityBenchmarkResult]:
    results: List[EntityBenchmarkResult] = []
    i = 1
    timeline_dataset_inputs = list(itertools.product(entity_names, processors))
    sklearn_dataset_inputs = list(itertools.product(dicterizers, clfs))
    for entity_name, processor in timeline_dataset_inputs:
        timeline_dataset = focals_to_timeline_dataset(focals, entity_name, processor)
        for dicterizer, clf in sklearn_dataset_inputs:
            sklearn_dataset = timeline_to_sklearn_dataset(timeline_dataset, dicterizer)
            scores = cross_val_score(clf(), sklearn_dataset.X, sklearn_dataset.y, cv=3)
            results.append(EntityBenchmarkResult(entity_name=entity_name,
                                                 preprocessor=processor.__name__,
                                                 dicterizer=dicterizer.__name__,
                                                 clf=str(clf),
                                                 scores=scores,
                                                 score_avg=statistics.mean(scores),
                                                 score_std=statistics.stdev(scores)))
            print(f'Benchmark iteration: {i} / {len(timeline_dataset_inputs) * len(sklearn_dataset_inputs)}')
            i += 1
    return results


def main():
    database = Database()
    focals = database.get_focals()
    references = list(database.get_averagely_popular_references(precision=1))
    entity_names = list(map(lambda r: r.name, references))
    results = benchmark(focals, entity_names, processors=[filter_and_slice_to_most_recent],
                        dicterizers=[counting_dicterizer], clfs=[DecisionTreeClassifier])
    pprint(results)
    summary_avg = statistics.mean(map(lambda r: r.score_avg, results))
    summary_std = statistics.mean(map(lambda r: r.score_std, results))
    print('Summary avg: ' + str(summary_avg))
    print('Summary std: ' + str(summary_std))


if __name__ == '__main__':
    main()
