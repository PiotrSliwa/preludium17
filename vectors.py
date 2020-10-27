from datetime import datetime
from math import floor

from sklearn.feature_extraction import DictVectorizer


class FeatureVectors:

    @staticmethod
    def __calculate_vector(feature_intensities_batch_dict):
        """
        :param feature_intensities_batch_dict: feature_intensities[key] = [(feature, intensity)], e.g. focal_features['@User'] = [('feature_A', 0.5), ('feature_B', 1)]
        :return: dict[key] = vector
        """
        vectorizer = DictVectorizer()
        vectorizer.fit(feature_intensities_batch_dict.values())
        return dict(map(lambda e: (e[0], vectorizer.transform(e[1])), feature_intensities_batch_dict.items()))

    @staticmethod
    def batched(reference_flows_batch_dict, feature_intensity_model):
        feature_intensities = {key: feature_intensity_model(flow) for key, flow in reference_flows_batch_dict.items()}
        return FeatureVectors.__calculate_vector(feature_intensities)

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
                                  reference_flows.values()))
            self.timestamp_min = min(list(map(lambda x: x[0], time_spans)))
            self.timestamp_max = max(list(map(lambda x: x[1], time_spans)))

        @staticmethod
        def __reference_timestamp(reference):
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

        @staticmethod
        def __flatten(reference_bucket_flow):
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


class CutToLastReferenceFeatureVectorsFactory:
    def __init__(self, reference_id, reference_flows_by_focal):
        self.cut_reference_flows = self.__cut_to_reference_id(reference_id, reference_flows_by_focal)

    @staticmethod
    def __index_of(matcher, arr):
        for index, elem in enumerate(arr):
            if matcher(elem):
                return index
        return None

    @staticmethod
    def __cut_to_reference_id(reference_id, reference_flows_by_focal):
        filtered_flow = {}
        for focal in reference_flows_by_focal:
            flow = reference_flows_by_focal[focal]
            try:
                last_index_of_reference = CutToLastReferenceFeatureVectorsFactory.__index_of(
                    lambda x: x['reference'] == reference_id, flow[::-1])
                filtered_reference_flow = flow[:last_index_of_reference]
            except ValueError:
                # Use full history when there is no global reference in there
                filtered_reference_flow = flow

            # Exclude the currently investigated reference
            filtered_flow[focal] = list(
                filter(lambda x: x['reference'] != reference_id, filtered_reference_flow))
        return filtered_flow

    def create(self, feature_intensity_model):
        return FeatureVectors.batched(self.cut_reference_flows, feature_intensity_model)