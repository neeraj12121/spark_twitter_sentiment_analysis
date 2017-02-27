import tweepy
import jsonpickle
from tweepy.streaming import StreamListener
import json

c_key = '......'
c_secret = '......'
token_key = '......'
token_secret = '......'

auth = tweepy.OAuthHandler(c_key,c_secret)
auth.set_access_token(token_key,token_secret)

api = tweepy.API(auth)

totalTweets=1000
fName='tweets.json'
tweetCount = 0


with open(fName, 'w') as f:
    while tweetCount < maxTweets:
        results = api.search(q='pepsi')
        if not results:
            print("No more tweets found")
            break
        for tweet in results:
            f.write(jsonpickle.encode(tweet._json, unpicklable=False) + '\n')
        tweetCount += len(results)
        print("Downloaded {0} tweets".format(tweetCount))

print ("Total {0} tweets are downloaded, Saved to {1}".format(tweetCount, fName))


class stream(StreamListener):
    def on_data(self, data):
	fhOut.write(data)
	j=json.loads(data)
	text=j["text"] 	
	print(text)

    def on_error(self, status):
	print("ERROR")
	print(status)

