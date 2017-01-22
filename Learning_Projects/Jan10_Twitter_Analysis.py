# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 14:02:40 2017

@author: sannadurai
"""

#Bag of words
#Sentiment Value of Tweet
#pip install tweepy
#pip install textblob
#signup apps from https://apps.twitter.com

#from textblob import TextBlob
#wiki = TextBlob("Siva is really good in Database and Data Science")
#print(wiki.tags)
#print(wiki.words)
#print(wiki.sentiment.polarity)

import tweepy
from textblob import TextBlob

consumer_key = 'xx'
consumer_secret = 'xx'

access_token = '-xx'
access_token_secret = 'cc'

auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)

api = tweepy.API(auth)
public_tweets = api.search('Retail')
for tweet in public_tweets:
    print(tweet)
    analysis = TextBlob(tweet.text)
    print(analysis.sentiment)

#Sentiment Analysis - Understand and extract feelings from data
#API lets you to access apps and functionality from code
#TextBlob is awesome for NLP tasks

