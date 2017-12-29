#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import logging.handlers

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s]: %(levelname)s: %(message)s')

import os
import re
import json
import sys
import time
import pandas as pd
import ftfy
import common
import csv
import codecs
from datetime import datetime
import nltk
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import pytz

import pprint
pp = pprint.PrettyPrinter(indent=4)


import nltk
from nltk.stem import WordNetLemmatizer
pattern=r'[A-Z]\.+\S+|\w+\-\w+|\w+'

def load_stoplist():
    stoplist = set()
    with open('./english.stop.txt', newline='', encoding='utf-8') as f:
        for line in f:
            stoplist.add(line.strip())
    return stoplist


def extract_clean_text(json_file):
    wv = []
    cnt = 0
    # stoplist = load_stoplist()
    wordnet_lemmatizer = WordNetLemmatizer()
    with open(json_file, 'r') as json_file:
        user_tweets = json.load(json_file)
        for user in user_tweets:
            text = ''
            for tweet in user_tweets[user]:
                text += common.cleanhtml(common.remove_hashtag_sign(common.remove_username(common.remove_url(ftfy.fix_text(tweet))))) + ' '
            clean_texts = [wordnet_lemmatizer.lemmatize(word.lower()) for word in nltk.regexp_tokenize(text, pattern)]
            wv.append(clean_texts)
            cnt += 1
    logger.info('total tweets: %d;'%cnt)
    return wv

def to_txt(wv, output_file):
    with open(output_file, 'w') as txt_file:
        for tweet in wv:
            for word in tweet:
                txt_file.write(word + ' ')
            txt_file.write('\n')

def extract_tweet_not_by_uid(source):
    tweets = []
    BTM_input = []
    wordnet_lemmatizer = WordNetLemmatizer()
    df = pd.read_csv(source,encoding='utf-8')
    for index, row in df.iterrows():
        clean_text = common.cleanhtml(common.remove_hashtag_sign(common.remove_username(common.remove_url(ftfy.fix_text(row['text'])))))
        preprocessed_text = ''
        temp = [wordnet_lemmatizer.lemmatize(word.lower()) for word in nltk.regexp_tokenize(clean_text, pattern)]
        if len(temp) == 0:
            continue
        for word in temp:
            preprocessed_text += word + ' '
        tweets.append(row['text'])
        BTM_input.append(preprocessed_text)

    with open('./intermediate_data/hpv_tweets/hpv_tweets_not_by_uid.csv', 'a', newline='', encoding='utf-8') as csv_f:
            writer = csv.DictWriter(csv_f, fieldnames=['tweets'], delimiter=',', quoting=csv.QUOTE_ALL)
            writer.writeheader()
            for tweet in tweets:
                writer.writerow({
                    'tweets': tweet})
    with open('./intermediate_data/hpv_tweets/hpv_tweets_not_by_uid_BTM_input.txt', 'w', encoding='utf-8') as outfile:
        for tweet in BTM_input:
            outfile.write(tweet + '\n')


fieldnames = ['clean_text', 'us_state','preprocessed_text','date']
def to_csv(results, csv_output_file):
        with open(csv_output_file, 'a', newline='', encoding='utf-8') as csv_f:
            writer = csv.DictWriter(csv_f, fieldnames=fieldnames, delimiter=',', quoting=csv.QUOTE_ALL)
            writer.writeheader()
            for tweet in results:
                writer.writerow({
                    'clean_text': tweet['clean_text'],
                    'us_state': tweet['us_state'],
                    'date': tweet['date'],
                    'preprocessed_text': tweet['preprocessed_text']})

def extract_text_for_BTM_topic_distribution(source, output_file):
    tweets = []
    stoplist = load_stoplist()
    wordnet_lemmatizer = WordNetLemmatizer()
    df = pd.read_csv(source,encoding='utf-8')
    for index, row in df.iterrows():
        tweet = {}
        clean_text = common.cleanhtml(common.remove_hashtag_sign(common.remove_username(common.remove_url(ftfy.fix_text(row['text'])))))
        preprocessed_text = ''
        temp = [wordnet_lemmatizer.lemmatize(word.lower()) for word in nltk.regexp_tokenize(clean_text, pattern) if wordnet_lemmatizer.lemmatize(word.lower()) not in stoplist]
        if len(temp) == 0:
            continue
        for word in temp:
            preprocessed_text += word + ' '
        date = datetime.strptime(row['created_at'],'%a %b %d %H:%M:%S +0000 %Y').replace(tzinfo=pytz.UTC)
        y_m = str(date.year) + '-' + (str(date.month) if (len(str(date.month)) == 2) else ('0'+str(date.month)))
        tweet['clean_text'] = clean_text
        tweet['us_state'] = row['us_state']
        tweet['preprocessed_text'] = preprocessed_text
        tweet['date'] = y_m
        tweets.append(tweet)
    logger.info(len(tweets))
    to_csv(tweets,output_file)


def transfer_csv_txt(csv_f,txt_file):
    txt = []
    df = pd.read_csv(csv_f, encoding='utf-8')
    with open(txt_file, 'w', encoding='utf-8') as outfile:
        for index, row in df.iterrows():
            outfile.write(row['preprocessed_text'] + '\n')

if __name__ == "__main__":

    logger.info(sys.version)

    # extract tweets not group by uid
    # extract_tweet_not_by_uid('./intermediate_data/hpv_geotagged.csv')


    # extract tweets by uid
    # wv = extract_clean_text('./intermediate_data/userid_list.json')
    # to_txt(wv, './intermediate_data/hpv_tweets/hpv_tweets.txt')

    # extract clean text, state info and preprocessed text
    # extract_text_for_BTM_topic_distribution('./intermediate_data/hpv_geotagged.csv', './intermediate_data/preprocessed_text_and_geo_date.csv')

    #transfer csv to txt
    # transfer_csv_txt('./intermediate_data/analysis/BTM/cutoffline_annotation/random_100.csv', './intermediate_data/analysis/BTM/cutoffline_annotation/random_100.txt')
    # transfer_csv_txt('./intermediate_data/preprocessed_text_and_geo.csv', './intermediate_data/preprocessed_text_and_geo.txt')