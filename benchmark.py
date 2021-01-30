import itertools
import statistics
from dataclasses import dataclass
from pprint import pprint
from typing import List

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from database import Database
from focals import Focal, FocalGroupSpan
from datasets import timeline_to_sklearn_dataset, Dicterizer, TimelineDataset
from dicterizers import counting_dicterizer
from processors import focals_to_timeline_dataset, TimelineProcessor, FilterAndSliceToMostRecentProcessor, TimepointProcessor


@dataclass(frozen=True)
class BenchmarkResult:
    processor: str
    dicterizer: str
    clf: str
    scores: List[float]
    score_avg: float
    score_std: float
    metrics: TimelineDataset.Metrics


def benchmark(focals: List[Focal],
              processors: List[TimelineProcessor],
              dicterizers: List[Dicterizer],
              clfs: List) -> List[BenchmarkResult]:
    results: List[BenchmarkResult] = []
    i = 1
    sklearn_dataset_inputs = list(itertools.product(dicterizers, clfs))
    for processor in processors:
        timeline_dataset = focals_to_timeline_dataset(focals, processor)
        for dicterizer, clf in sklearn_dataset_inputs:
            sklearn_dataset = timeline_to_sklearn_dataset(timeline_dataset, dicterizer)
            scores = cross_val_score(clf(), sklearn_dataset.X, sklearn_dataset.y, cv=sklearn_dataset.splits)
            results.append(BenchmarkResult(processor=str(processor),
                                           dicterizer=dicterizer.__name__,
                                           clf=str(clf),
                                           scores=scores,
                                           score_avg=statistics.mean(scores) if len(scores) > 1 else scores[0],
                                           score_std=statistics.stdev(scores) if len(scores) > 1 else 0,
                                           metrics=timeline_dataset.metrics()))
            print(f'Benchmark iteration: {i} / {len(processors) * len(sklearn_dataset_inputs)}')
            i += 1
    return results


def main():
    database = Database()
    focals = database.get_focals()
    focal_group_span = FocalGroupSpan(focals)
    highest_distribution_point = focal_group_span.highest_distribution_points()[0]
    print(f'Highest distribution point: {highest_distribution_point}')
    references = list(database.get_averagely_popular_references(precision=1))
    entity_names = [r.name for r in references]
    processors = [FilterAndSliceToMostRecentProcessor(entity_name) for entity_name in entity_names] + [
        TimepointProcessor(entity_name, highest_distribution_point.timepoint) for entity_name in entity_names]
    results = benchmark(focals, processors=processors, dicterizers=[counting_dicterizer], clfs=[DecisionTreeClassifier])
    pprint(results)
    summary_avg = statistics.mean((r.score_avg for r in results))
    summary_std = statistics.mean((r.score_std for r in results))
    print('Summary avg: ' + str(summary_avg))
    print('Summary std: ' + str(summary_std))


if __name__ == '__main__':
    main()
