# %% DB

from database import get_local_database, get_reference_flows

db = get_local_database()
reference_flows = get_reference_flows(db)

# %%

from sklearn.feature_extraction import DictVectorizer
from scipy.spatial import distance
from itertools import combinations


def get_references():
    most_popular_reference = next(db.materialized_reference_popularity.find().sort([('popularity', -1)]))
    max_popularity = most_popular_reference['popularity']
    reference_popularity = db.materialized_reference_popularity.find(
        {'popularity': {'$lte': int(max_popularity / 2)}}).sort([('popularity', -1), ('_id', 1)])
    return reference_popularity


def calculate_vectors(reference_id, calculate_feature_intensities):
    vectorizer = DictVectorizer()
    features = {}
    for focal in reference_flows:
        reference_flow = reference_flows[focal]
        try:
            last_index_of_reference = len(reference_flow) - reference_flow[::-1].index(reference_id) - 1
            limited_reference_flow = reference_flow[:last_index_of_reference]
        except ValueError:
            # Use full history when there is no global reference in there
            limited_reference_flow = reference_flow

        # Exclude the currently investigated reference
        limited_reference_flow = list(filter(lambda x: x != reference_id, limited_reference_flow))

        # Feature selection algorithm (key is the feature and value is the intensity)
        features[focal] = calculate_feature_intensities(limited_reference_flow)

    vectorizer.fit(features.values())
    return dict(map(lambda e: (e[0], vectorizer.transform(e[1])), features.items()))


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

    non_scoped_focals = list(filter(lambda x: x not in scoped_focals, vectors.keys()))
    return {
        'scoped': average_distance(scoped_focals),
        'non_scoped': average_distance(non_scoped_focals),
        'between': average_distance_between(scoped_focals, non_scoped_focals)
    }


import pandas as pd


class FeatureIntensitiesModel:

    @staticmethod
    def mere_occurrence(reference_flow):
        return dict([(reference, 1) for reference in reference_flow])

    @staticmethod
    def count_occurrences(reference_flow):
        result = {}
        for reference in reference_flow:
            current_count = result.setdefault(reference, 0)
            result[reference] = current_count + 1
        return result

    @staticmethod
    def linear_fading_occurences(reference_flow):
        result = {}
        step = 1.0 / len(reference_flow)
        intensity = step
        for reference in reference_flow:
            current_intensity = result.setdefault(reference, 0.0)
            result[reference] = current_intensity + intensity
            intensity += step
        return result


def test_distances(reference, features_intensities_model):
    vectors = calculate_vectors(reference['_id'], features_intensities_model)
    distances = calculate_distances(vectors, reference['focals'])
    return {
        **distances,
        'reference': reference['_id'],
        'supports_hypothesis': distances['scoped'] < distances['non_scoped'],
        'relative_difference': (distances['non_scoped'] - distances['scoped']) / distances['non_scoped']
    }


class FeaturesIntensitiesBenchmark:
    runs = {}

    def run(self, reference, features_intensities_model):
        model_name = features_intensities_model.__name__
        result = test_distances(reference, features_intensities_model)
        self.runs.setdefault(model_name, []).append(result)

    def print_summary(self):
        for model_name in self.runs:
            run = self.runs[model_name]
            df = pd.DataFrame(run)
            print(f'model_name: {model_name}')
            print(df['supports_hypothesis'].value_counts())
            print(f'Mean: {df["relative_difference"].mean()}')
            print(f'Std: {df["relative_difference"].std()}')


references = get_references()
benchmark = FeaturesIntensitiesBenchmark()
i = 0
for reference in references:
    print('==========')
    print(i)
    benchmark.run(reference, FeatureIntensitiesModel.mere_occurrence)
    benchmark.run(reference, FeatureIntensitiesModel.count_occurrences)
    benchmark.run(reference, FeatureIntensitiesModel.linear_fading_occurences)
    benchmark.print_summary()
    i += 1
