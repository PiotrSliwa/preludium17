import argparse

import GetOldTweets3 as got

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
        print(f'Processed {current_len - previous_len} tweet(s). Total: {len(self.processed_tweet_ids)}')


def collect(username):
    tweet_collection = TweetCollection(username)
    tweet_criteria = got.manager.TweetCriteria().setUsername(username)
    got.manager.TweetManager.getTweets(tweet_criteria,
                                       debug=False,
                                       # bufferLength=1,
                                       receiveBuffer=tweet_collection.add_all)
    return tweet_collection.processed_tweet_ids


def most_popular_referenced_users():
    db = get_local_database()
    candidates = db.reference_popularity.find({'_id': {'$regex': '^@'}}).sort('popularity', -1)
    tweets = db.tweets
    for candidate in candidates:
        username = candidate['_id'].replace('@', '')
        tweet = tweets.find_one({'username': username})
        if tweet is None:
            yield username

def get(username):
    print(f'Getting tweets of user {username}')
    processed_tweet_ids = collect(username)
    print(f'Finished successfully. Processed {len(processed_tweet_ids)} tweets.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retrieve user tweets.')
    parser.add_argument('action', nargs=1, help='list (lists the most popular referenced user without tweets in the database), get (retrieves tweets of a specified user), next (get tweets of the first user from the list)')
    parser.add_argument('user', nargs='?')
    args = parser.parse_args()
    action = args.action[0]
    if action == 'get':
        if args.user is None:
            parser.print_help()
        else:
            username = args.user
            get(username)
    elif action == 'list':
        users = most_popular_referenced_users()
        for i in range(3):
            print(next(users))
    elif action == 'next':
        users = most_popular_referenced_users()
        username = next(users)
        get(username)
    else:
        parser.print_help()