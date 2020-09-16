import argparse

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


class Database:
    db = get_local_database()

    def most_popular_nodes(self, num=1):
        collection = self.db['most_popular_nodes']
        return collection.find(sort=[('knownByCount', -1)], limit=num)


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
