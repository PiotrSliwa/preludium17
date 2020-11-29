# %%

from pprint import pprint

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

from database import get_local_database, get_reference_flows_by_focal, get_current_users
from vectors import FeatureVectors, CutToLastReferenceFeatureVectorsFactory


def get_scoped_focals(db, reference_id):
    reference_popularity = next(db.materialized_reference_popularity.find({'_id': reference_id}))
    return reference_popularity['focals']


def get_balanced_reference_ids(db, range=0):
    current_users = get_current_users(db)
    number_of_users = len(current_users)
    balanced_reference_popularities = db.materialized_reference_popularity.find(
        {'popularity': {'$gte': number_of_users / 2 - range, '$lte': number_of_users / 2 + range}})
    for balanced_reference_popularity in balanced_reference_popularities:
        yield balanced_reference_popularity['_id']


def get_dataset(db, reference_id, feature_intensity_model):
    scoped_focals = get_scoped_focals(db, reference_id)
    feature_vectors_factory = CutToLastReferenceFeatureVectorsFactory(reference_id, reference_flows_by_focal)
    feature_vectors = feature_vectors_factory.create(feature_intensity_model)

    X = []
    y = []
    for focal, vector in feature_vectors.items():
        X.append(vector.toarray()[0])
        y.append(1 if focal in scoped_focals else 0)

    return (X, y)


db = get_local_database()
reference_ids = get_balanced_reference_ids(db, range=2)
reference_flows_by_focal = get_reference_flows_by_focal(db)
temporal_intensities_model = FeatureVectors.TemporalIntensitiesModel(reference_flows_by_focal)
feature_intensity_models = [FeatureVectors.StaticFeatureIntensitiesModel.mere_occurrence,
                            FeatureVectors.StaticFeatureIntensitiesModel.count_occurrences,
                            temporal_intensities_model.linear_fading_summing]
clf_inits = [RandomForestClassifier,
             tree.DecisionTreeClassifier]
scoring = {'acc': 'accuracy',
           'f1': 'f1'}
db.classifier_benchmarks.drop()
for reference_id in reference_ids:
    for feature_intensity_model in feature_intensity_models:
        for clf_init in clf_inits:
            print(f'\n*** {reference_id} / {feature_intensity_model.__name__} / {clf_init.__name__}')
            (X, y) = get_dataset(db, reference_id, feature_intensity_model)
            clf = clf_init()
            scores = cross_validate(clf, X, y, scoring=scoring, cv=5)
            pprint(scores)
            db.classifier_benchmarks.insert_one({
                'reference_id': reference_id,
                'feature_intensity_model': feature_intensity_model.__name__,
                'clf_name': clf_init.__name__,
                'test_acc_mean': scores["test_acc"].mean(),
                'test_acc_std': scores["test_acc"].std(),
                'test_f1_mean': scores["test_f1"].mean(),
                'test_f1_std': scores["test_f1"].std()
            })
            print(f'acc: mean {scores["test_acc"].mean()}, std {scores["test_acc"].std()}')
            print(f'f1: mean {scores["test_f1"].mean()}, std {scores["test_f1"].std()}')
