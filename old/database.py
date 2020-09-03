from pymongo import MongoClient


def get_local_database():
    client = MongoClient('mongodb://localhost:27017/')
    return client['preludium']


class Database:
    db = get_local_database()

    def most_popular_nodes(self, num=1):
        collection = self.db['most_popular_nodes']
        return collection.find(sort=[('knownByCount', -1)], limit=num)