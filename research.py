# %% DB

from database import get_local_database, get_reference_flows

db = get_local_database()
reference_flows = get_reference_flows(db)

# %%

from sklearn.feature_extraction import DictVectorizer
from scipy.spatial import distance
from itertools import combinations
import csv
import os
from datetime import datetime


def get_references():
    most_popular_reference = next(db.materialized_reference_popularity.find().sort([('popularity', -1)]))
    max_popularity = most_popular_reference['popularity']
    reference_popularity = db.materialized_reference_popularity.find(
        {'popularity': {'$lte': int(max_popularity / 2)}}).sort([('popularity', -1), ('_id', 1)])
    return reference_popularity


def calculate_vectors(feature_intensities):
    """
    :param feature_intensities: feature_intensities[focal] = [(feature, intensity)], e.g. focal_features['@User'] = [('feature_A', 0.5), ('feature_B', 1)]
    :return: dict[focal] = vector
    """
    vectorizer = DictVectorizer()
    vectorizer.fit(feature_intensities.values())
    return dict(map(lambda e: (e[0], vectorizer.transform(e[1])), feature_intensities.items()))


def index_of(matcher, arr):
    for index, elem in enumerate(arr):
        if matcher(elem):
            return index
    return None


def filter_reference_flows(reference_id):
    filtered_reference_flows = {}
    for focal in reference_flows:
        reference_flow = reference_flows[focal]
        try:
            last_index_of_reference = index_of(lambda x: x['reference'] == reference_id, reference_flow[::-1])
            filtered_reference_flow = reference_flow[:last_index_of_reference]
        except ValueError:
            # Use full history when there is no global reference in there
            filtered_reference_flow = reference_flow

        # Exclude the currently investigated reference
        filtered_reference_flows[focal] = list(filter(lambda x: x['reference'] != reference_id, filtered_reference_flow))
    return filtered_reference_flows


def calculate_feature_intensities(filtered_reference_flows, feature_intensities_model):
    feature_intensities = {}
    for focal in filtered_reference_flows:
        filtered_reference_flow = filtered_reference_flows[focal]
        # Feature selection algorithm (key is the feature and value is the intensity)
        feature_intensities[focal] = feature_intensities_model(filtered_reference_flow)
    return feature_intensities


def get_non_scoped_focals(vectors, scoped_focals):
    return list(filter(lambda x: x not in scoped_focals, vectors.keys()))


def calculate_distances(vectors, scoped_focals):
    def focal_distance(focal_a, focal_b):
        return distance.euclidean(vectors[focal_a].toarray(), vectors[focal_b].toarray())

    def average_distance(focals):
        total = 0.0
        count = 0
        for (a, b) in combinations(focals, 2):
            total += focal_distance(a, b)
            count += 1
        return total / count

    def average_distance_between(focals_a, focals_b):
        total = 0.0
        count = 0
        for a in focals_a:
            for b in focals_b:
                total += focal_distance(a, b)
                count += 1
        return total / count

    non_scoped_focals = get_non_scoped_focals(vectors, scoped_focals)
    return {
        'scoped': average_distance(scoped_focals),
        'non_scoped': average_distance(non_scoped_focals),
        'between': average_distance_between(scoped_focals, non_scoped_focals)
    }


import pandas as pd


class StaticFeatureIntensitiesModel:
    @staticmethod
    def mere_occurrence(reference_flow):
        return dict([(reference['reference'], 1) for reference in reference_flow])

    @staticmethod
    def count_occurrences(reference_flow):
        result = {}
        for reference in reference_flow:
            reference_id = reference['reference']
            current_count = result.setdefault(reference_id, 0)
            result[reference_id] = current_count + 1
        return result


class TemporalIntensitiesModel:
    def __init__(self):
        time_spans = list(map(lambda x: [self.__reference_timestamp(x[0]), self.__reference_timestamp(x[-1])], reference_flows.values()))
        self.timestamp_min = min(list(map(lambda x: x[0], time_spans)))
        self.timestamp_max = max(list(map(lambda x: x[1], time_spans)))

    def __reference_timestamp(self, reference):
        date = reference['date']
        if isinstance(date, datetime):
            return datetime.timestamp(date)
        elif isinstance(date, str):
            return datetime.timestamp(datetime.strptime(reference['date'], '%Y-%m-%d %H:%M:%S'))
        else:
            raise ValueError

    def __intensity(self, reference):
        timestamp = self.__reference_timestamp(reference)
        return (timestamp - self.timestamp_min) / (self.timestamp_max - self.timestamp_min)

    def linear_fading_summing(self, reference_flow):
        result = {}
        for reference in reference_flow:
            reference_id = reference['reference']
            current_intensity = result.setdefault(reference_id, 0.0)
            result[reference_id] = current_intensity + self.__intensity(reference)
        return result


class DistanceBenchmark:
    runs = []

    @staticmethod
    def convert(string):
        try:
            return float(string)
        except ValueError as e:
            return string

    def read_csv(self):
        result = []
        if not os.path.exists(self.csv_file):
            return result
        with open(self.csv_file, newline='') as f:
            reader = csv.DictReader(f)
            line_count = 0
            for row in reader:
                if line_count > 0:
                    converted_row = {k: DistanceBenchmark.convert(v) for k, v in row.items()}
                    result.append(converted_row)
                line_count += 1
        return result

    def write_csv(self):
        with open(self.csv_file, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=['scoped', 'non_scoped', 'between', 'reference', 'supports_hypothesis', 'relative_difference', 'model_name'])
            writer.writeheader()
            writer.writerows(self.runs)

    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.runs = self.read_csv()

    def contains(self, reference):
        for run in self.runs:
            if run['reference'] == reference:
                return True
        return False

    def run(self, reference_popularity, features_intensities_models):
        reference = reference_popularity['_id']
        if self.contains(reference):
            print(f'Ignoring {reference}')
            return
        filtered_reference_flows = filter_reference_flows(reference)
        for model in features_intensities_models:
            model_name = model.__name__
            feature_intensities = calculate_feature_intensities(filtered_reference_flows, model)
            vectors = calculate_vectors(feature_intensities)
            distances = calculate_distances(vectors, reference_popularity['focals'])
            result = {
                **distances,
                'reference': reference,
                'supports_hypothesis': distances['scoped'] < distances['non_scoped'],
                'relative_difference': (distances['non_scoped'] - distances['scoped']) / distances['non_scoped'],
                'model_name': model_name
            }
            self.runs.append(result)

    def summary(self):
        df = pd.DataFrame(self.runs)
        print('Mean:')
        print(df.groupby('model_name')['relative_difference'].mean())
        print('Std:')
        print(df.groupby('model_name')['relative_difference'].std())
        self.write_csv()


class ClassificationBenchmark:
    runs = {}

    # train classifier

    def run(self, reference, features_intensities_models):
        pass

    def summary(self, csv_filename):
        pass


reference_popularities = get_references()
benchmark = DistanceBenchmark('out.csv')
i = 0
summary_batch = 1
temporal_intensities_model = TemporalIntensitiesModel()
for reference_popularity in reference_popularities:
    print('==========')
    print(i)
    benchmark.run(reference_popularity, [StaticFeatureIntensitiesModel.mere_occurrence,
                                         StaticFeatureIntensitiesModel.count_occurrences,
                                         temporal_intensities_model.linear_fading_summing])
    if i % summary_batch == 0:
        benchmark.summary()
    i += 1
