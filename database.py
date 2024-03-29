import argparse
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Iterator

import numpy
from pymongo import MongoClient

from focals import Focal
from timelines import EntityName, Reference


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


@dataclass(frozen=True)
class ReferencePopularity:
    name: EntityName
    popularity: int


class Database:
    db = get_local_database()

    @staticmethod
    def __to_reference_popularity(doc) -> ReferencePopularity:
        return ReferencePopularity(doc['_id'], doc['popularity'])

    def get_focals(self) -> List[Focal]:
        docs = self.db.materialized_information_flow.find().sort('date', 1)
        result: Dict[EntityName, Focal] = {}
        for doc in docs:
            focal = doc['focal']
            reference = doc['reference']
            raw_date = doc['date']
            date = raw_date if type(raw_date) == datetime else datetime.strptime(raw_date, '%Y-%m-%d %H:%M:%S')
            reference_flow = result.setdefault(focal, Focal(name=focal, timeline=[]))
            reference_flow.timeline.append(Reference(name=reference, date=date))
            result[focal] = reference_flow
        return list(result.values())

    def get_most_popular_reference(self) -> ReferencePopularity:
        docs = self.get_most_popular_references()
        return next(docs)

    def get_most_popular_references(self) -> Iterator[ReferencePopularity]:
        docs = self.db.materialized_reference_popularity.find().sort('popularity', -1)
        return (self.__to_reference_popularity(doc) for doc in docs)

    def get_averagely_popular_references(self, precision=5) -> Iterator[ReferencePopularity]:
        most_popular = self.get_most_popular_reference()
        average_popularity = most_popular.popularity / 2
        docs = self.db.materialized_reference_popularity.find(
            {'popularity': {'$gte': average_popularity - precision, '$lte': average_popularity + precision}})
        return map(self.__to_reference_popularity, docs)

    @staticmethod
    def to_dict(obj):
        if hasattr(obj, '__dict__'):
            return {i: Database.to_dict(v) for i, v in obj.__dict__.items()}
        elif type(obj) is dict:
            return obj
        elif hasattr(obj, '__iter__') and type(obj) is not str:
            return [Database.to_dict(v) for v in obj]
        elif type(obj) is int or type(obj) is float or type(obj) is str or type(obj) is numpy.float64:
            return obj
        else:
            return str(obj)

    def save(self, collection_name: str, results: List, clean=True):
        collection = self.db[collection_name]
        if clean:
            collection.drop()
        dict_results = Database.to_dict(results)
        collection.insert_many(dict_results)

    def drop(self, collection_name: str):
        collection = self.db[collection_name]
        collection.drop()


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
