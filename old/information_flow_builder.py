from pymongo import MongoClient
from pprint import pprint

client = MongoClient('mongodb://localhost:27017/?readPreference=primary&appname=MongoDB%20Compass&ssl=false')
result = client['preludium']['tweets'].aggregate([
    {
        '$addFields': {
            'to': {
                '$concatArrays': [
                    {
                        '$split': [
                            '$hashtags', ' '
                        ]
                    }, {
                        '$split': [
                            '$mentions', ' '
                        ]
                    }, {
                        '$split': [
                            '$urls', ','
                        ]
                    }
                ]
            }, 
            'from': {
                '$concat': [
                    '@', '$username'
                ]
            }
        }
    }, {
        '$unwind': {
            'path': '$to'
        }
    }, {
        '$match': {
            '$expr': {
                '$ne': [
                    '$to', ''
                ]
            }
        }
    }, {
        '$project': {
            'from': 1, 
            'to': 1, 
            'date': 1
        }
    }
])

for r in result:
    pprint(r)