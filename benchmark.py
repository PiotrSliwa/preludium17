import itertools
import json
import statistics
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Dict, Callable, Any

from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from database import Database
from datasets import timeline_to_sklearn_dataset, Dicterizer, TimelineDataset
from dicterizers import counting_dicterizer
from focals import Focal, FocalGroupSpan
from processors import focals_to_timeline_dataset, TimelineProcessor, FilterAndSliceToMostRecentProcessor, WindowingProcessor


@dataclass(frozen=True)
class BenchmarkResult:
    processor: Dict
    dicterizer: str
    classifier: str
    scores: List[float]
    score_avg: float
    score_std: float
    metrics: TimelineDataset.Metrics


ClassifierFactory = Callable[[], Any]


def benchmark(focals: List[Focal],
              processors: List[TimelineProcessor],
              dicterizers: List[Dicterizer],
              classifier_factories: List[ClassifierFactory]) -> List[BenchmarkResult]:
    results: List[BenchmarkResult] = []
    i = 1
    sklearn_dataset_inputs = list(itertools.product(dicterizers, classifier_factories))
    for processor in processors:
        timeline_dataset = focals_to_timeline_dataset(focals, processor)
        for dicterizer, classifier_factory in sklearn_dataset_inputs:
            sklearn_dataset = timeline_to_sklearn_dataset(timeline_dataset, dicterizer, shuffle_classes=False)
            classifier = classifier_factory()
            scores = cross_val_score(classifier, sklearn_dataset.X, sklearn_dataset.y, cv=sklearn_dataset.splits)
            results.append(
                BenchmarkResult(processor={**Database.to_dict(processor), 'type': processor.__class__.__name__},
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


def filter(benchmark_results: List[BenchmarkResult], test_to_training_min_value,
           test_class_ratio_max_divergence) -> FilteredBenchmarkResults:
    filtered_results = FilteredBenchmarkResults([], [])
    for result in benchmark_results:
        if abs(result.metrics.test_class_ratio - 0.5) <= test_class_ratio_max_divergence and result.metrics.test_to_training_ratio >= test_to_training_min_value:
            filtered_results.accepted.append(result)
        else:
            filtered_results.off_limits.append(result)
    return filtered_results


def to_json(object) -> str:
    return json.dumps(object.__dict__, indent=4, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o))


def mlp_classifier():
    return MLPClassifier()


def main():
    database = Database()
    focals = database.get_focals()
    focal_group_span = FocalGroupSpan(focals)
    highest_distribution_point = focal_group_span.highest_distribution_points()[0]
    print(f'Highest distribution point: {highest_distribution_point}')
    most_popular_references = database.get_most_popular_references()
    references = [next(most_popular_references) for i in range(500)]
    # references = list(database.get_averagely_popular_references(precision=15))
    database.drop('research_references')
    database.save('research_references', references)
    entity_names = [r.name for r in references]
    processors = [WindowingProcessor(entity_name, highest_distribution_point.timepoint, timedelta(weeks=weeks)) for entity_name in entity_names for weeks in (8, 16, 24)]
    # processors = [FilterAndSliceToMostRecentProcessor('@forzegg'), FilterAndSliceToMostRecentProcessor('#TBT')]
    # processors = [FilterAndSliceToMostRecentProcessor(entity_name) for entity_name in entity_names] + [TimepointProcessor(entity_name, highest_distribution_point.timepoint) for entity_name in entity_names] + [SlicingProcessor(entity_name, highest_distribution_point.timepoint) for entity_name in entity_names]
    dicterizers = [counting_dicterizer]
    classifier_factories = [DecisionTreeClassifier]
    results = benchmark(focals, processors, dicterizers, classifier_factories)
    test_to_training_min_value = 0.2
    test_class_ratio_max_divergence = 0.2
    filtered_results = filter(results, test_to_training_min_value, test_class_ratio_max_divergence)
    print(
        f'Filtered results test_to_training_min_value: {test_to_training_min_value}, test_class_ratio_max_divergence: {test_class_ratio_max_divergence}')
    print(to_json(filtered_results))
    if len(filtered_results.accepted) > 0:
        summary_avg = statistics.mean((r.score_avg for r in filtered_results.accepted))
        summary_std = statistics.mean((r.score_std for r in filtered_results.accepted))
        print('Summary avg: ' + str(summary_avg))
        print('Summary std: ' + str(summary_std))
        database.save('results_accepted', filtered_results.accepted)
        print('Accepted results saved.')
    else:
        database.drop('results_accepted')
        print('No accepted results.')
    if len(filtered_results.off_limits) > 0:
        database.save('results_off_limits', filtered_results.off_limits)
        print('Off-limits results saved.')
    else:
        database.drop('results_off_limits')
        print('No off-limits results.')
    database.drop('results_all')
    database.save('results_all', results)


if __name__ == '__main__':
    main()
