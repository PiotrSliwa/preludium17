from math import ceil, floor
from database import get_local_database, get_reference_flows_by_focal
from sklearn.feature_extraction import DictVectorizer
from scipy.spatial import distance
from itertools import combinations
from datetime import datetime
import concurrent.futures
import time


class ReferenceFlows:
    def __init__(self, db):
        self.reference_flows_by_focal = get_reference_flows_by_focal(db)

    def get_cut_to_reference_id(self, reference_id):
        filtered_flow = {}
        for focal in self.reference_flows_by_focal:
            flow = self.reference_flows_by_focal[focal]
            try:
                last_index_of_reference = index_of(lambda x: x['reference'] == reference_id, flow[::-1])
                filtered_reference_flow = flow[:last_index_of_reference]
            except ValueError:
                # Use full history when there is no global reference in there
                filtered_reference_flow = flow

            # Exclude the currently investigated reference
            filtered_flow[focal] = list(
                filter(lambda x: x['reference'] != reference_id, filtered_reference_flow))
        return filtered_flow

    def get_all(self):
        return self.reference_flows_by_focal


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


def calculate_feature_intensities(filtered_reference_flows, feature_intensities_model):
    feature_intensities = {}
    for focal in filtered_reference_flows:
        filtered_reference_flow = filtered_reference_flows[focal]
        # Feature selection algorithm (key is the feature and value is the intensity)
        feature_intensities[focal] = feature_intensities_model(filtered_reference_flow)
    return feature_intensities


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
    n_buckets = 10

    def __init__(self, reference_flows):
        time_spans = list(map(lambda x: [self.__reference_timestamp(x[0]), self.__reference_timestamp(x[-1])],
                              reference_flows.get_all().values()))
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

    def __relative_time(self, reference):
        """
        :param reference:
        :return: Relative time in the global timeframe (i.e. "0" when it is the first reference in the analyzed dataset, "1" if the last, "0.5" if in the middle, etc.)
        """
        timestamp = self.__reference_timestamp(reference)
        return float(timestamp - self.timestamp_min) / float(self.timestamp_max - self.timestamp_min)

    def __bucket_id(self, reference):
        """
        :param reference:
        :return: A bucket id of the given reference
        """
        bucket_id = floor(self.__relative_time(reference) * self.n_buckets)
        if bucket_id < self.n_buckets:
            return bucket_id
        return self.n_buckets - 1

    def __flatten(self, reference_bucket_flow):
        result = {}
        for reference_id, buckets in reference_bucket_flow.items():
            for bucket_id, value in enumerate(buckets):
                result[f'{reference_id}__{bucket_id}'] = value
        return result

    def linear_fading_summing(self, reference_flow):
        result = {}
        for reference in reference_flow:
            reference_id = reference['reference']
            current_intensity = result.setdefault(reference_id, 0.0)
            result[reference_id] = current_intensity + self.__relative_time(reference)
        return result

    def count_occurrences_in_time_buckets(self, reference_flow):
        result = {}
        for reference in reference_flow:
            reference_id = reference['reference']
            current_buckets = result.setdefault(reference_id, [0] * self.n_buckets)
            bucket_id = self.__bucket_id(reference)
            current_count = current_buckets[bucket_id]
            current_buckets[bucket_id] = current_count + 1
        return self.__flatten(result)


class DistanceBenchmark:
    runs = []
    executor = concurrent.futures.ThreadPoolExecutor()

    @staticmethod
    def convert(string):
        try:
            return float(string)
        except ValueError as e:
            return string

    def __read(self):
        cursor = self.results_collection.find({})
        for document in cursor:
            yield document

    def __init__(self, db, reference_flows, fresh_start):
        self.results_collection = db.distance_benchmarks
        self.reference_flows = reference_flows
        if fresh_start:
            self.results_collection.drop()
        self.runs = list(self.__read())

    def contains(self, reference_id):
        for run in self.runs:
            if run['reference_id'] == reference_id:
                return True
        return False

    def __distance(self, vector_a, vector_b):
        return distance.euclidean(vector_a.toarray(), vector_b.toarray())

    def __average_distance(self, vectors):
        total = 0.0
        count = 0
        focals = vectors.keys()
        for (a, b) in combinations(focals, 2):
            total += self.__distance(vectors[a], vectors[b])
            count += 1
        return total / count

    def __split(self, vectors, scoped_focals):
        scoped = dict(filter(lambda x: x[0] in scoped_focals, vectors.items()))
        non_scoped = dict(filter(lambda x: x[0] not in scoped_focals, vectors.items()))
        return scoped, non_scoped

    def __run_single_model(self, reference_id, reference_popularity, cut_reference_flows, model):
        start_time = time.time()
        model_name = model.__name__
        feature_intensities = calculate_feature_intensities(cut_reference_flows, model)
        vectors = calculate_vectors(feature_intensities)
        scoped_vectors, non_scoped_vectors = self.__split(vectors, reference_popularity['focals'])
        avg_distance_scoped = self.__average_distance(scoped_vectors)
        avg_distance_non_scoped = self.__average_distance(non_scoped_vectors)
        duration = time.time() - start_time
        result = {
            'scoped': avg_distance_scoped,
            'non_scoped': avg_distance_non_scoped,
            'reference_id': reference_id,
            'supports_hypothesis': avg_distance_scoped < avg_distance_non_scoped,
            'relative_difference': (avg_distance_non_scoped - avg_distance_scoped) / avg_distance_non_scoped,
            'model_name': model_name,
            'scoped_focals': ' '.join(scoped_vectors.keys()),
            'non_scoped_focals': ' '.join(non_scoped_vectors.keys()),
            'duration': duration
        }
        self.results_collection.replace_one({'reference': reference_id, 'model_name': model_name}, result, True)
        self.runs.append(result)
        return f'model {model} finished in {duration}s'

    def run(self, reference_popularity, features_intensities_models):
        reference_id = reference_popularity['_id']
        if self.contains(reference_id):
            print(f'Ignoring {reference_id}')
            return
        cut_reference_flows = self.reference_flows.get_cut_to_reference_id(reference_id)
        futures = []
        for model in features_intensities_models:
            futures.append(self.executor.submit(self.__run_single_model,
                                                reference_id=reference_id,
                                                reference_popularity=reference_popularity,
                                                cut_reference_flows=cut_reference_flows,
                                                model=model))
        for future in concurrent.futures.as_completed(futures):
            print(future.result())


class ClassificationBenchmark:
    runs = {}

    # train classifier

    def run(self, reference, features_intensities_models):
        pass

    def summary(self, csv_filename):
        pass


if __name__ == '__main__':
    db = get_local_database()
    reference_popularities = get_references()
    reference_flows = ReferenceFlows(db)
    benchmark = DistanceBenchmark(db, reference_flows, fresh_start=True)
    temporal_intensities_model = TemporalIntensitiesModel(reference_flows)
    i = 0
    for reference_popularity in reference_popularities:
        print(f'\n====================\n{i}. {reference_popularity["_id"]}')
        benchmark.run(reference_popularity, [StaticFeatureIntensitiesModel.mere_occurrence,
                                             StaticFeatureIntensitiesModel.count_occurrences,
                                             temporal_intensities_model.linear_fading_summing,
                                             temporal_intensities_model.count_occurrences_in_time_buckets])
        i += 1
