# Import api keys and secerets
import keys
# import libraries
import tweepy
import pandas as pd
import numpy as np


# Authentication
def twitterAuth():
    consumerKey = keys.API_KEY
    consumerSecret = keys.API_KEY_SECRET
    accessToken = keys.API_ACCESS_TOKEN
    accessTokenSecret = keys.API_ACCESS_TOKEN_SECRET

    try:
        auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
        auth.set_access_token(accessToken, accessTokenSecret)
        api = tweepy.API(auth)
    except:
        print("Authentication failed")

    return api


def cleanTweet(text):
    return 0


def getTweets(query, numberOfTweets):
    tweets = []
    fetched_tweets = api.search_tweets(
        q=query, count=numberOfTweets, lang="en")
    # for tweet in fetched_tweets:
    #     cleanTweet(tweet.text)
    return fetched_tweets


# Get access to api
api = twitterAuth()

# Set query and number of tweets to get
query = "Elon Musk"
numberOfTweets = 1
tweets = getTweets(query, numberOfTweets)
