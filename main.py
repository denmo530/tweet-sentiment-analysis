# Import api keys and secerets
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
import re
import numpy as np
import pandas as pd
import tweepy
import keys
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.datasets import load_files
stopWords = set(stopwords.words("english"))
# import libraries
# Authentication

# Read dataset
data = pd.read_csv("dataset.csv", encoding="latin", header=None)
data.columns = ['Sentiment', 'Id', 'Date', 'Query', 'User_name', 'Text']
# Drop unnecessary columns, we only want the sentiment and text
data = data.drop(['Id', 'Date', 'Query', 'User_name'], axis=1)
data = data.sample(10000)
# data = data[data.Text.notnull()]

sentiment = {0: "Negative", 4: "Positive"}


def sentimentDecoder(label):
    return sentiment[label]


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
    # Remove
    text = re.sub(
        "(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|[0-9]", "", text)
    # Lowercase letters
    text = text.lower()
    # Split sentences into words
    text = text.split()
    # Remove stopwords
    text = [x for x in text if not x in stopWords]
    # Stem words to base form, cats to cat, running to run etc
    text = [ps.stem(word) for word in text]
    # Join all the split words again to form sentences
    text = " ".join(text)
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


def vectorize(tweets):
    vectorizer = TfidfVectorizer(
        max_features=2500, min_df=7, max_df=0.8, stop_words="english")
    # fit and tranform using training text
    processed_features = vectorizer.fit_transform(tweets).toarray()

    return processed_features


# Get access to api
api = twitterAuth()
# Set query and number of tweets to get
query = "#Nature"
numberOfTweets = 100
tweets = getTweets(query, numberOfTweets)
data.Sentiment = data.Sentiment.apply(lambda x: sentimentDecoder(x))
corpus = []
corpus = data.Text.apply(cleanTweet)
processed_features = vectorize(corpus)
# Split training and test data
x_train, x_test, y_train, y_test = train_test_split(
    processed_features, data.Sentiment, test_size=0.2, random_state=0)


# Train data using ML classifier
# Random forest
text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier.fit(x_train, y_train)
RandomForestClassifier(bootstrap=True, class_weight=None, criterion="gini", max_depth=None, max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0.0,
                       min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=None, oob_score=False, random_state=0, verbose=0, warm_start=False)
predictions = text_classifier.predict(x_test)

print("Random Forest Classifier: \n")
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print(accuracy_score(y_test, predictions))
print("\n")

# Naive Bayes
print("Naive Bayes classifier \n")
nb = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
nb.fit(x_train, y_train)
y_pred_class = nb.predict(x_test)
print(accuracy_score(y_test, y_pred_class))
print(confusion_matrix(y_test, y_pred_class))
print("\n")
