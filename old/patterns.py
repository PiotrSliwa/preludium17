# Create a database client

from old.database import get_local_database
db = get_local_database()
tf30 = db['leading_information_30days']

## Compare all leading information between two nodes using 30-day timeframe.
# To do so, let's first try to aggregate all leading information by Mentioning->Mentioned and see if there are similarities between each Mentioning node within one Mentioned node.

import pandas as pd
import time

start_time = time.process_time()
aggregated = pd.DataFrame(columns=['from', 'to', 'lead'])
docs = tf30.find()
for doc in docs:
    for li in doc['leading_information']:
        aggregated = aggregated.append({
            'from': doc['from'], 
            'to': doc['to'],
            'lead': li['to']
        }, ignore_index=True)

print(f'finished in {time.process_time() - start_time}')