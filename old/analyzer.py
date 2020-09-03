import json
import copy
import pandas as pd


def to_information_flow_edge(tweet):
    return {
        'from': 'twitter-user:@' + tweet['username'],
        'to': tweet['entity'],
        'date': tweet['date']
    }


def load_tweets(name):
    with open(f'{name}.json') as f:
        contents = f.read()
        return json.loads(contents)


def tagged_entities(label, names):
    filtered_names = list(filter(lambda name: len(name) > 0, names))
    return list(map(lambda name: f'{label}:{name}', filtered_names))


def aggregate_entities(tweet):
    hashtags = tweet['hashtags'].split(' ')
    mentions = tweet['mentions'].split(' ')
    urls = tweet['urls'].split(',')
    tweet_copy = copy.copy(tweet)
    tweet_copy['entities'] = tagged_entities('twitter-hashtag', hashtags) + tagged_entities('twitter-user', mentions) + tagged_entities('url', urls)
    return tweet_copy


def spread_by_entities(tweets):
    spread_tweets = []
    for tweet in tweets:
        for entity in tweet['entities']:
            new_tweet = copy.copy(tweet)
            new_tweet['entity'] = entity
            spread_tweets.append(new_tweet)
    return spread_tweets


def save_information_flow_edges(name, edges):
    with open(f'{name}-ife.json', 'w') as f:
        contents = json.dumps(edges, indent=4)
        f.write(contents)



name = 'dagadi'
tweets = load_tweets(name)
tweets = list(map(aggregate_entities, tweets))
tweets = spread_by_entities(tweets)
edges = list(map(to_information_flow_edge, tweets))
edges_df = pd.DataFrame(edges)
save_information_flow_edges(name, edges)
print('end')