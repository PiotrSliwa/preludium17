import pandas as pd
from old.database import get_local_database

collection = 'information_flow_edges'

db = get_local_database()
tweets = db[collection]
df = pd.DataFrame(list(tweets.find()))
df.to_csv(f'{collection}.csv')
print('done')