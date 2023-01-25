# Import api keys and secerets
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import metrics
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

sentiment = {0: "Negative", 2: "Neutral", 4: "Positive"}


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
    fetched_tweets = api.search_tweets(
        q=query, count=numberOfTweets, lang="en", tweet_mode="extended")

    return fetched_tweets


def vectorize(tweets, vectorizer):
    # transform tweet data
    vectorized_tweets = vectorizer.transform(tweets).toarray()

    return vectorized_tweets


data.Sentiment = data.Sentiment.apply(lambda x: sentimentDecoder(x))
corpus = []
corpus = data.Text.apply(cleanTweet)
vectorizer = TfidfVectorizer(
    max_features=2500, min_df=7, max_df=0.8, stop_words="english")
processed_features = vectorizer.fit_transform(corpus).toarray()

# Split training and test data
x_train, x_test, y_train, y_test = train_test_split(
    processed_features, data.Sentiment, test_size=0.2, random_state=0)

# Gridsearch parameters
parameters = {
    'n_estimators': [10, 50, 100],
    'max_depth': [2, 5, 10]
}

parameters_nb = {'alpha': [0.1, 1, 10]}

parameters_svm = {'loss': ['hinge', 'log_loss', 'modified_huber'],
                  'penalty': ['l2', 'l1', 'elasticnet'],
                  'alpha': [1e-2, 1e-3]}


grid_search = GridSearchCV(RandomForestClassifier(),
                           parameters, scoring="accuracy", n_jobs=-1)
grid_search.fit(x_train, y_train)
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
grid_predictions = best_model.predict(x_test)

print("Grid Search: Random forest \n")
print("Best score: ", grid_search.best_score_)
print("Best parameters:", best_params)
print("Best Model: ", best_model)
print(classification_report(y_test, grid_predictions))
print(confusion_matrix(y_test, grid_predictions))
print("Accuracy:", accuracy_score(y_test, grid_predictions))
print("\n")

grid_nb = GridSearchCV(MultinomialNB(), parameters_nb,
                       scoring="accuracy", n_jobs=-1)
grid_nb.fit(x_train, y_train)
best_params_nb = grid_nb.best_params_
best_model_nb = grid_nb.best_estimator_
grid_predictions_nb = best_model_nb.predict(x_test)

print("Grid Search: Naive Bayes \n")
print("Best score: ", grid_nb.best_score_)
print("Best parameters:", grid_nb.best_params_)
print("Best Model: ", grid_nb.best_estimator_)
print(classification_report(y_test, grid_predictions_nb))
print(confusion_matrix(y_test, grid_predictions_nb))
print("Accuracy:", accuracy_score(y_test, grid_predictions_nb))
print("\n")

grid_svm = GridSearchCV(SGDClassifier(), parameters_svm,
                        scoring="accuracy", n_jobs=-1)
grid_svm.fit(x_train, y_train)
best_params_svm = grid_svm.best_params_
best_model_svm = grid_svm.best_estimator_
grid_predictions_svm = best_model_svm.predict(x_test)

print("Grid Search: SVM \n")
print("Best score: ", grid_svm.best_score_)
print("Best parameters:", best_params_svm)
print("Best Model: ", grid_svm.best_estimator_)
print(classification_report(y_test, grid_predictions_svm))
print(confusion_matrix(y_test, grid_predictions_svm))
print("Accuracy:", accuracy_score(y_test, grid_predictions_svm))
print("\n")

# Use the best model of the naive bayes search on tweets from twitter
# Get access to api
api = twitterAuth()
# Set query and number of tweets to get
query = "google"
numberOfTweets = 100
tweets = getTweets(query, numberOfTweets)
processed_tweets = []
for tweet in tweets:
    try:
        text = cleanTweet(tweet.retweeted_status.full_text)
        processed_tweets.append(text)
    except:
        text = cleanTweet(tweet.full_text)
        processed_tweets.append(text)

vectorized_tweets = vectorize(processed_tweets, vectorizer)
sentiment_predictions = best_model_nb.predict(vectorized_tweets)

# Zip the tweets and predictions together
tweet_predictions = list(zip(processed_tweets, sentiment_predictions))
tweet_sentiment = pd.DataFrame.from_records(
    tweet_predictions, columns=["tweet", "sentiment"])
# Match original tweet with prediction
for i, tweet in enumerate(tweets):
    if hasattr(tweet, 'retweeted_status'):
        original_tweet = tweet.retweeted_status.full_text
    else:
        original_tweet = tweet.full_text
    sentiment = sentiment_predictions[i]
    tweet_sentiment.loc[i] = [original_tweet, sentiment]
    # Print tweet and prediction for each tweet
print("\n###################\n")

for index, row in tweet_sentiment.iterrows():

    print("Tweet:", row["tweet"], "\n", "Sentiment:", row["sentiment"])
    print("\n")
# for tweet in tweet_sentiment:
#     try:
#         print(f"Retweet: {tweet.retweeted_status.full_text}")
#         print(f"Prediction: {sentiment_predictions}")
#         print("\n")
#     except:
#         print(f"Tweet: {tweet.full_text}")
#         print(f"Prediction: {sentiment_predictions}")
#         print("\n")
print("\n###################\n")

# Train data using ML classifier
# Random forest
# text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
# text_classifier.fit(x_train, y_train)
# RandomForestClassifier(bootstrap=True, class_weight=None, criterion="gini", max_depth=None, max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0.0,
#                        min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=None, oob_score=False, random_state=0, verbose=0, warm_start=False)
# predictions = text_classifier.predict(x_test)

# print("Random Forest Classifier: \n")
# print(confusion_matrix(y_test, predictions))
# print(classification_report(y_test, predictions))
# print(accuracy_score(y_test, predictions))
# print("\n")

# # Naive Bayes
# print("Naive Bayes classifier \n")
# nb = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
# nb.fit(x_train, y_train)
# y_pred_class = nb.predict(x_test)
# print(accuracy_score(y_test, y_pred_class))
# print(confusion_matrix(y_test, y_pred_class))
# print("\n")

# print("SVM Classifier \n")
# svm = SGDClassifier(loss="hinge", penalty="l2", alpha=1e-3,
#                     random_state=42, max_iter=5, tol=None)
# svm.fit(x_train, y_train)
# svm_predicted = svm.predict(x_test)
# print("SVM accuracy ", np.mean(svm_predicted == y_test))
# print(classification_report(y_test, svm_predicted))
