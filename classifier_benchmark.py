#%%

from sklearn import svm
from sklearn.model_selection import cross_val_score

from database import get_local_database, get_reference_flows_by_focal
from vectors import FeatureVectors, CutToLastReferenceFeatureVectorsFactory


def get_scoped_focals(db, reference_id):
    reference_popularity = next(db.materialized_reference_popularity.find({'_id': reference_id}))
    return reference_popularity['focals']


db = get_local_database()
reference_ids = ['@FallGuysGame', '#Dreamhack', '#DHS17', '@eswc_en']
reference_flows_by_focal = get_reference_flows_by_focal(db)
temporal_intensities_model = FeatureVectors.TemporalIntensitiesModel(reference_flows_by_focal)
for reference_id in reference_ids:
    print(reference_id)

    scoped_focals = get_scoped_focals(db, reference_id)
    non_scoped_focals = list(filter(lambda x: x not in scoped_focals, reference_flows_by_focal.keys()))
    feature_vectors_factory = CutToLastReferenceFeatureVectorsFactory(reference_id, reference_flows_by_focal)
    feature_vectors = feature_vectors_factory.create(temporal_intensities_model.linear_fading_summing)

    X = []
    y = []
    for focal, vector in feature_vectors.items():
        X.append(vector.toarray()[0])
        y.append(1 if focal in scoped_focals else 0)

    clf = svm.SVC()
    scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
    print(scores)
