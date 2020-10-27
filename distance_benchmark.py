import concurrent.futures
import time
from itertools import combinations

from scipy.spatial import distance

from database import get_local_database, get_reference_flows_by_focal
from vectors import FeatureVectors, CutToLastReferenceFeatureVectorsFactory


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

    @staticmethod
    def __distance(vector_a, vector_b):
        return distance.euclidean(vector_a.toarray(), vector_b.toarray())

    def __average_distance(self, vectors):
        total = 0.0
        count = 0
        focals = vectors.keys()
        for (a, b) in combinations(focals, 2):
            total += self.__distance(vectors[a], vectors[b])
            count += 1
        return total / count

    @staticmethod
    def __split(vectors, scoped_focals):
        scoped = dict(filter(lambda x: x[0] in scoped_focals, vectors.items()))
        non_scoped = dict(filter(lambda x: x[0] not in scoped_focals, vectors.items()))
        return scoped, non_scoped

    def __run_single_model(self, reference_id, scoped_focals, feature_vectors_factory, model):
        start_time = time.time()
        model_name = model.__name__
        vectors = feature_vectors_factory.create(model)
        scoped_vectors, non_scoped_vectors = self.__split(vectors, scoped_focals)
        avg_distance_scoped = self.__average_distance(scoped_vectors)
        avg_distance_non_scoped = self.__average_distance(non_scoped_vectors)
        avg_distance_all = self.__average_distance(vectors)
        duration = time.time() - start_time
        result = {
            'all': avg_distance_all,
            'scoped': avg_distance_scoped,
            'non_scoped': avg_distance_non_scoped,
            'reference_id': reference_id,
            'supports_hypothesis': avg_distance_scoped < avg_distance_all,
            'relative_difference': (avg_distance_all - avg_distance_scoped) / avg_distance_all,
            'model_name': model_name,
            'scoped_focals': ' '.join(scoped_vectors.keys()),
            'non_scoped_focals': ' '.join(non_scoped_vectors.keys()),
            'duration': duration
        }
        self.results_collection.replace_one({'reference': reference_id, 'model_name': model_name}, result, True)
        self.runs.append(result)
        return f'model {model} finished in {duration}s'

    def run(self, reference_id, scoped_focals, features_intensities_models):
        if self.contains(reference_id):
            print(f'Ignoring {reference_id}')
            return
        feature_vectors_factory = CutToLastReferenceFeatureVectorsFactory(reference_id, self.reference_flows)
        futures = []
        for model in features_intensities_models:
            futures.append(self.executor.submit(self.__run_single_model,
                                                reference_id=reference_id,
                                                scoped_focals=scoped_focals,
                                                feature_vectors_factory=feature_vectors_factory,
                                                model=model))
        for future in concurrent.futures.as_completed(futures):
            print(future.result())


def get_reference_popularities():
    most_popular_reference = next(db.materialized_reference_popularity.find().sort([('popularity', -1)]))
    max_popularity = most_popular_reference['popularity']
    reference_popularity = db.materialized_reference_popularity.find(
        {'popularity': {'$lte': int(max_popularity / 2)}}).sort([('popularity', -1), ('_id', 1)])
    return reference_popularity


if __name__ == '__main__':
    db = get_local_database()
    reference_popularities = get_reference_popularities()
    reference_flows_by_focal = get_reference_flows_by_focal(db)
    benchmark = DistanceBenchmark(db, reference_flows_by_focal, fresh_start=True)
    temporal_intensities_model = FeatureVectors.TemporalIntensitiesModel(reference_flows_by_focal)
    i = 0
    for reference_popularity in reference_popularities:
        reference_id = reference_popularity['_id']
        scoped_focals = reference_popularity['focals']
        print(f'\n====================\n{i}. {reference_id}')
        benchmark.run(reference_id,
                      scoped_focals,
                      features_intensities_models=[FeatureVectors.StaticFeatureIntensitiesModel.mere_occurrence,
                                                   FeatureVectors.StaticFeatureIntensitiesModel.count_occurrences,
                                                   temporal_intensities_model.linear_fading_summing,
                                                   temporal_intensities_model.count_occurrences_in_time_buckets])
        i += 1
