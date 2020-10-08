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


def calculate_vectors(feature_intensities):
    """
    :param feature_intensities: feature_intensities[focal] = [(feature, intensity)], e.g. focal_features['@User'] = [('feature_A', 0.5), ('feature_B', 1)]
    :return: dict[focal] = vector
    """
    vectorizer = DictVectorizer()
    vectorizer.fit(feature_intensities.values())
    return dict(map(lambda e: (e[0], vectorizer.transform(e[1])), feature_intensities.items()))


def limit_reference_flows(reference_id):
    limited_reference_flows = {}
    for focal in reference_flows:
        reference_flow = reference_flows[focal]
        try:
            last_index_of_reference = len(reference_flow) - reference_flow[::-1].index(reference_id) - 1
            limited_reference_flow = reference_flow[:last_index_of_reference]
        except ValueError:
            # Use full history when there is no global reference in there
            limited_reference_flow = reference_flow

        # Exclude the currently investigated reference
        limited_reference_flows[focal] = list(filter(lambda x: x != reference_id, limited_reference_flow))
    return limited_reference_flows


def calculate_feature_intensities(limited_reference_flows, feature_intensities_model):
    feature_intensities = {}
    for focal in limited_reference_flows:
        limited_reference_flow = limited_reference_flows[focal]
        # Feature selection algorithm (key is the feature and value is the intensity)
        feature_intensities[focal] = feature_intensities_model(limited_reference_flow)
    return feature_intensities


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
    def linear_fading_summing(reference_flow):
        result = {}
        step = 1.0 / len(reference_flow)
        intensity = step
        for reference in reference_flow:
            current_intensity = result.setdefault(reference, 0.0)
            result[reference] = current_intensity + intensity
            intensity += step
        return result

    @staticmethod
    def linear_fading_most_recent(reference_flow):
        result = {}
        step = 1.0 / len(reference_flow)
        intensity = step
        for reference in reference_flow:
            result[reference] = intensity
            intensity += step
        return result

    @staticmethod
    def smoothstep_fading_summing(reference_flow):
        result = {}
        step = 1.0 / len(reference_flow)
        x = step
        for reference in reference_flow:
            current_intensity = result.setdefault(reference, 0.0)
            result[reference] = current_intensity + 3 * x ** 2 - 2 * x ** 2
            x += step
        return result

    @staticmethod
    def smoothstep_fading_most_recent(reference_flow):
        result = {}
        step = 1.0 / len(reference_flow)
        x = step
        for reference in reference_flow:
            result[reference] = 3 * x ** 2 - 2 * x ** 2
            x += step
        return result


class FeaturesIntensitiesBenchmark:
    runs = {}

    def run(self, reference, features_intensities_models):
        limited_reference_flows = limit_reference_flows(reference['_id'])
        for model in features_intensities_models:
            model_name = model.__name__
            feature_intensities = calculate_feature_intensities(limited_reference_flows, model)
            vectors = calculate_vectors(feature_intensities)
            distances = calculate_distances(vectors, reference['focals'])
            result = {
                **distances,
                'reference': reference['_id'],
                'supports_hypothesis': distances['scoped'] < distances['non_scoped'],
                'relative_difference': (distances['non_scoped'] - distances['scoped']) / distances['non_scoped']
            }
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
    benchmark.run(reference, [FeatureIntensitiesModel.mere_occurrence, FeatureIntensitiesModel.count_occurrences,
                              FeatureIntensitiesModel.linear_fading_summing,
                              FeatureIntensitiesModel.linear_fading_most_recent,
                              FeatureIntensitiesModel.smoothstep_fading_summing,
                              FeatureIntensitiesModel.smoothstep_fading_most_recent])
    benchmark.print_summary()
    i += 1
