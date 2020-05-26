import GetOldTweets3 as got
import json
import sys


def save(username, tweets):
    tweets_json = to_json(tweets)
    with open(f'{username}.json', 'w') as f:
        f.write(tweets_json)


class TweetCollection:
    def __init__(self, username):
        self.username = username
        self.tweets = {}

    def add_all(self, tweets):
        previous_len = len(self.tweets)
        for tweet in tweets:
            self.tweets[tweet.id] = tweet.__dict__
        current_len = len(self.tweets)
        if previous_len == current_len:
            raise Exception('No new tweets added!')
        save(self.username, self.get_all())

    def get_all(self):
        return list(self.tweets.values())


def collect(username):
    tweet_collection = TweetCollection(username)
    tweet_criteria = got.manager.TweetCriteria().setUsername(username)
    got.manager.TweetManager.getTweets(tweet_criteria,
                                       debug=True,
                                       bufferLength=1,
                                       receiveBuffer=tweet_collection.add_all)
    return tweet_collection.get_all()


def to_json(tweets):
    return json.dumps(tweets,
                      indent=4,
                      sort_keys=True,
                      default=str)


if __name__ == '__main__':
    username = 'PCGamesN'
    tweets = collect(username)
    print('Finished successfully')