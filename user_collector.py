import GetOldTweets3 as got
from pymongo import MongoClient
from database import get_local_database


class TweetCollection:
    def __init__(self, username):
        self.username = username
        self.processed_tweet_ids = set()
        self.db = get_local_database()
        self.db_collection = self.db.tweets

    def save(self, tweet):
        doc = {**tweet, '_id': tweet['id']}
        self.db_collection.replace_one({'_id': doc['_id']}, doc, True)

    def add_all(self, tweets):
        previous_len = len(self.processed_tweet_ids)
        for tweet in tweets:
            tweet_dict = tweet.__dict__
            self.processed_tweet_ids.add(tweet_dict['id'])
            self.save(tweet_dict)
        current_len = len(self.processed_tweet_ids)
        if previous_len == current_len:
            raise Exception('No new tweets added!')


def collect(username):
    tweet_collection = TweetCollection(username)
    tweet_criteria = got.manager.TweetCriteria().setUsername(username)
    got.manager.TweetManager.getTweets(tweet_criteria,
                                       debug=True,
                                       bufferLength=1,
                                       receiveBuffer=tweet_collection.add_all)
    return tweet_collection.processed_tweet_ids


if __name__ == '__main__':
    username = 'heroicgg'
    processed_tweet_ids = collect(username)
    print(f'Finished successfully. Processed {len(processed_tweet_ids)} tweets.')