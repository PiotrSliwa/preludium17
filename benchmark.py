import itertools
import json
import statistics
from dataclasses import dataclass
from pprint import pprint
from typing import List, Dict

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from database import Database
from focals import Focal, FocalGroupSpan
from datasets import timeline_to_sklearn_dataset, Dicterizer, TimelineDataset
from dicterizers import counting_dicterizer
from processors import focals_to_timeline_dataset, TimelineProcessor, FilterAndSliceToMostRecentProcessor, TimepointProcessor, SlicingProcessor


@dataclass(frozen=True)
class BenchmarkResult:
    processor: Dict
    dicterizer: str
    classifier: str
    scores: List[float]
    score_avg: float
    score_std: float
    metrics: TimelineDataset.Metrics


def benchmark(focals: List[Focal],
              processors: List[TimelineProcessor],
              dicterizers: List[Dicterizer],
              classifier_inits: List) -> List[BenchmarkResult]:
    results: List[BenchmarkResult] = []
    i = 1
    sklearn_dataset_inputs = list(itertools.product(dicterizers, classifier_inits))
    for processor in processors:
        timeline_dataset = focals_to_timeline_dataset(focals, processor)
        for dicterizer, classifier_init in sklearn_dataset_inputs:
            sklearn_dataset = timeline_to_sklearn_dataset(timeline_dataset, dicterizer, shuffle_classes=False)
            classifier = classifier_init()
            scores = cross_val_score(classifier, sklearn_dataset.X, sklearn_dataset.y, cv=sklearn_dataset.splits)
            results.append(BenchmarkResult(processor=processor.to_dict(),
                                           dicterizer=dicterizer.__name__,
                                           classifier=str(classifier),
                                           scores=scores,
                                           score_avg=statistics.mean(scores) if len(scores) > 1 else scores[0],
                                           score_std=statistics.stdev(scores) if len(scores) > 1 else 0,
                                           metrics=timeline_dataset.metrics()))
            print(f'Benchmark iteration: {i} / {len(processors) * len(sklearn_dataset_inputs)}')
            i += 1
    return results


@dataclass
class FilteredBenchmarkResults:
    accepted: List[BenchmarkResult]
    off_limits: List[BenchmarkResult]


def filter(benchmark_results: List[BenchmarkResult], test_to_training_ratio_delta, class_ratio_delta) -> FilteredBenchmarkResults:
    filtered_results = FilteredBenchmarkResults([], [])
    for result in benchmark_results:
        if abs(result.metrics.test_class_ratio - 0.5) <= class_ratio_delta and abs(
                result.metrics.training_class_ratio - 0.5) <= class_ratio_delta and abs(
                result.metrics.test_to_training_ratio - 0.5) <= test_to_training_ratio_delta:
            filtered_results.accepted.append(result)
        else:
            filtered_results.off_limits.append(result)
    return filtered_results


def to_json(object) -> str:
    return json.dumps(object.__dict__, indent=4, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o))


def main():
    database = Database()
    focals = database.get_focals()
    focal_group_span = FocalGroupSpan(focals)
    highest_distribution_point = focal_group_span.highest_distribution_points()[0]
    print(f'Highest distribution point: {highest_distribution_point}')
    references = list(database.get_averagely_popular_references(precision=1))
    entity_names = [r.name for r in references]
    processors = [FilterAndSliceToMostRecentProcessor(entity_name) for entity_name in entity_names]
    # processors = [FilterAndSliceToMostRecentProcessor(entity_name) for entity_name in entity_names] + [TimepointProcessor(entity_name, highest_distribution_point.timepoint) for entity_name in entity_names] + [SlicingProcessor(entity_name, highest_distribution_point.timepoint) for entity_name in entity_names]
    results = benchmark(focals,
                        processors=processors,
                        dicterizers=[counting_dicterizer],
                        classifier_inits=[DecisionTreeClassifier])
    test_to_training_ratio_delta = 0.3
    class_ratio_delta = 0.3
    filtered_results = filter(results, test_to_training_ratio_delta, class_ratio_delta)
    print(f'Filtered results test_to_training_ratio_delta: {test_to_training_ratio_delta}, class_ratio_delta: {class_ratio_delta}')
    print(to_json(filtered_results))
    summary_avg = statistics.mean((r.score_avg for r in filtered_results.accepted))
    summary_std = statistics.mean((r.score_std for r in filtered_results.accepted))
    print('Summary avg: ' + str(summary_avg))
    print('Summary std: ' + str(summary_std))


if __name__ == '__main__':
    main()
