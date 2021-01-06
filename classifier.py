from typing import List

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from database import Database, Focal
from datasets import TimelineDataset, timeline_to_sklearn_dataset
from dicterizers import counting_dicterizer
from preprocessors import Preprocessor, EntityPreprocessor


def focals_to_timeline_dataset(focals: List[Focal], preprocessor: Preprocessor) -> TimelineDataset:
    result = TimelineDataset()
    for focal in focals:
        dataset = preprocessor(focal.timeline)
        result += dataset
    return result


def main():
    database = Database()
    focals = database.get_focals()
    entity_name = database.get_averagely_popular_reference().name
    entity_preprocessor = EntityPreprocessor(entity_name)
    timeline_dataset = focals_to_timeline_dataset(focals, entity_preprocessor.filter_and_slice_to_most_recent)
    sklearn_dataset = timeline_to_sklearn_dataset(timeline_dataset, counting_dicterizer)
    clf = DecisionTreeClassifier()
    print(cross_val_score(clf, sklearn_dataset.X, sklearn_dataset.y, cv=3))


if __name__ == '__main__':
    main()
