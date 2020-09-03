from pymongo import MongoClient
import pandas as pd

if __name__ == '__main__':
    client = MongoClient('mongodb://localhost:27017/')
    db = client['preludium']
    edges = db.information_flow_edges
    print('x')