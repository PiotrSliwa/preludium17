import argparse
from datetime import datetime
from typing import List, Dict, TypedDict, NamedTuple

from pymongo import MongoClient


def get_local_database():
    client = MongoClient('mongodb://localhost:27017/')
    return client['preludium']


def materialize_views():
    db = get_local_database()

    db.tweets.aggregate([
        {
            '$set': {
                'hashtags': {
                    '$split': [
                        '$hashtags', ' '
                    ]
                },
                'mentions': {
                    '$split': [
                        '$mentions', ' '
                    ]
                }
            }
        }, {
            '$set': {
                'reference': {
                    '$setUnion': [
                        '$hashtags', '$mentions', [
                            {
                                '$concat': [
                                    '@', '$to'
                                ]
                            }
                        ]
                    ]
                }
            }
        }, {
            '$project': {
                '_id': 1,
                'date': 1,
                'focal': {
                    '$concat': [
                        '@', '$username'
                    ]
                },
                'reference': '$reference'
            }
        }, {
            '$unwind': {
                'path': '$reference'
            }
        }, {
            '$match': {
                '$and': [
                    {
                        'reference': {
                            '$ne': ''
                        }
                    }, {
                        'reference': {
                            '$ne': '@'
                        }
                    }, {
                        'reference': {
                            '$ne': None
                        }
                    }
                ]
            }
        }, {
            '$set': {
                'tweet_id': '$_id',
                '_id': {
                    '$concat': [
                        '$_id', '$reference'
                    ]
                }
            }
        }, {
            '$out': 'materialized_information_flow'
        }
    ])

    db.materialized_information_flow.aggregate([
        {
            '$group': {
                '_id': '$reference',
                'focals': {
                    '$addToSet': '$focal'
                }
            }
        }, {
            '$set': {
                'popularity': {
                    '$size': '$focals'
                }
            }
        }, {
            '$out': 'materialized_reference_popularity'
        }
    ])


def get_reference_flows_by_focal(db):
    reference_flows = {}
    information_flow = db.materialized_information_flow.find().sort('date', 1)
    for row in information_flow:
        focal = row['focal']
        if not focal in reference_flows:
            reference_flows[focal] = []
        reference_flows[focal].append(row)
    return reference_flows


def get_current_users(db):
    aggregated = db.tweets.aggregate([{'$group': {'_id': '$username'}}])
    return list(map(lambda x: x['_id'], aggregated))


EntityId = str


class Reference(NamedTuple):
    id: EntityId
    date: datetime


class ReferenceFlow(NamedTuple):
    focal: EntityId
    references: List[Reference]


class Database:
    db = get_local_database()

    def __to_reference_flow(self, doc) -> ReferenceFlow:
        pass

    def get_reference_flows(self) -> List[ReferenceFlow]:
        docs = self.db.materialized_information_flow.find().sort('date', 1)
        result: Dict[EntityId, ReferenceFlow] = {}
        for doc in docs:
            focal = doc['focal']
            reference = doc['reference']
            date = doc['date']
            reference_flow = result.setdefault(focal, ReferenceFlow(focal=focal, references=[]))
            reference_flow.references.append(Reference(reference, date))
            result[focal] = reference_flow
        return list(result.values())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Manage the database.')
    parser.add_argument('action', nargs=1, help='materialize (materialized all views dependent on tweets)')
    args = parser.parse_args()
    action = args.action[0]
    if action == 'materialize':
        materialize_views()
        print('Done.')
    else:
        parser.print_help()
