import argparse
import re
from datetime import datetime

from snscrape.modules import twitter

from database import get_local_database, materialize_views


def log(msg):
    now = datetime.now()
    print(f'{now.strftime("%d/%m/%Y %H:%M:%S")} - {msg}')


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
            self.processed_tweet_ids.add(tweet['id'])
            if tweet['username'] != self.username:
                print(
                    f"Saving username {tweet['username']} while the collection is for {self.username}. Probably it is an alias. Add {self.username}to .usercollectorignore to mark it as done.")
            self.save(tweet)
        current_len = len(self.processed_tweet_ids)
        if previous_len == current_len:
            raise Exception('No new tweets added!')
        log(f'Processed {current_len - previous_len} tweet(s). Total: {len(self.processed_tweet_ids)}')


def scrape_hashtags(string):
    return re.findall(r'(#\w+)\b', string)


def normalize_tweet(tweet):
    return {
        "username": tweet.user.username.strip(),
        "to": '',
        "text": tweet.content.strip(),
        "retweets": tweet.retweetCount,
        "favorites": tweet.likeCount,
        "replies": tweet.replyCount,
        "id": str(tweet.id),
        "permalink": tweet.url.strip(),
        "author_id": '',
        "date": tweet.date,
        "formatted_date": tweet.date.isoformat(),
        "hashtags": (' '.join(scrape_hashtags(tweet.content.strip()))).strip(),
        "mentions": (' '.join(list(map(lambda x: '@' + x.username,
                                       tweet.mentionedUsers))) if tweet.mentionedUsers is not None else '').strip(),
        "geo": '',
        "urls": (' '.join(tweet.outlinks)).strip()
    }


def collect(username):
    tweet_collection = TweetCollection(username)
    tweets = twitter.TwitterUserScraper(username).get_items()
    for tweet in tweets:
        normalized_tweet = normalize_tweet(tweet)
        tweet_collection.add_all([normalized_tweet])
    return tweet_collection.processed_tweet_ids


def get_ignored_users():
    with open('.usercollectorignore') as f:
        return list(map(str.strip, f.readlines()))


def get_current_users(db):
    aggregated = db.tweets.aggregate([{'$group': {'_id': '$username'}}])
    return list(map(lambda x: x['_id'], aggregated))


def most_popular_referenced_users():
    db = get_local_database()
    candidates = db.materialized_reference_popularity.find({'_id': {'$regex': '^@'}}).sort('popularity', -1)
    ignored_users = get_ignored_users()
    current_users = get_current_users(db)
    start = datetime.now()
    for candidate in candidates:
        username = candidate['_id'].replace('@', '')
        if username in ignored_users or username in current_users:
            continue
        print(f'Yielded a candidate in {datetime.now() - start}')
        yield username


def get(username):
    log(f'Getting tweets of user {username}')
    processed_tweet_ids = collect(username)
    log(f'Finished successfully. Processed {len(processed_tweet_ids)} tweets.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retrieve user tweets.')
    parser.add_argument('action', nargs=1,
                        help='list (lists the most popular referenced user without tweets in the database), get (retrieves tweets of a specified user), next (get tweets of the first user from the list)')
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
        try:
            get(username)
        finally:
            print('Materializing views...')
            start = datetime.now()
            materialize_views()
            print(f'Finished materializing in {datetime.now() - start}')
    else:
        parser.print_help()
