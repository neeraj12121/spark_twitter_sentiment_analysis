c_key = '......'
c_secret = '......'
token_key = '......'
token_secret = '......'


auth = tweepy.OAuthHandler(c_key,c_secret)
auth.set_access_token(token_key,token_secret)

api = tweepy.API(auth)


