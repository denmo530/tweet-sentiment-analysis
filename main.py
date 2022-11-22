# Import api keys and secerets
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import keys
# import libraries
import tweepy
import pandas as pd
import numpy as np
import string
import re
import nltk
from nltk.tokenize import word_tokenize

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
    ps = PorterStemmer()
    stopWords = set(stopwords.words("english"))
    # Remove
    text = " ".join(re.sub(
        "(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())
    # Lowercase letters and replace - with space
    text = text.lower().replace("-", " ")
    # Remove numbers and stowords
    table = str.maketrans('', '', string.punctuation+string.digits)
    text = text.translate(table)
    # Tokenize words
    tokens = word_tokenize(text)
    # Stem words to base form, cats to cat, running to run etc
    text = [ps.stem(word) for word in tokens]
    # Remove the stopwords
    words = [word for word in text if not word in stopWords]
    # Join all the split words again to form sentences
    text = " ".join(words)
    return text


def getTweets(query, numberOfTweets):
    tweets = []
    fetched_tweets = api.search_tweets(
        q=query, count=numberOfTweets, lang="en", tweet_mode="extended")

    for tweet in fetched_tweets:
        try:
            text = cleanTweet(tweet.retweeted_status.full_text)
            tweets.append(text)
        except:
            text = cleanTweet(tweet.full_text)
            tweets.append(text)

    return tweets


# Get access to api
api = twitterAuth()

# Set query and number of tweets to get
query = "#Nature"
numberOfTweets = 100
tweets = getTweets(query, numberOfTweets)
for tweet in tweets:
    print(tweet)
