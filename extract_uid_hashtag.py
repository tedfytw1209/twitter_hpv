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
import common
import csv
import json
import pandas as pd
import ftfy


def extract_username(source):
    unique_userid = {}
    df = pd.read_csv(source,encoding='utf-8')
    for index, row in df.iterrows():
        if row['user_id'] not in unique_userid:
            unique_userid[row['user_id']] = [row['clean_text']]
        else:
            unique_userid[row['user_id']].append(row['clean_text'])
    return unique_userid

def calculate_tweets_per_user(path):
    cnt_user_one_tweets = 0
    with open(path) as json_data:
        unique_userid = json.load(json_data)
        n_tweets = []
        for user in unique_userid:
            if len(unique_userid[user]) == 1:
                cnt_user_one_tweets += 1
            n_tweets.append(len(unique_userid[user]))
        logger.info('max user tweets: %d;'%max(n_tweets))
        logger.info('min user tweets: %d;'%min(n_tweets))
        logger.info('user with only one tweets: %d;'%cnt_user_one_tweets)
        logger.info('user more than one tweets: %d;'%(99227-cnt_user_one_tweets))


def extract_hashtag(source):
    hashtag = {}
    cnt = 0
    df = pd.read_csv(source,encoding='utf-8')
    for index, row in df.iterrows():
        if '#' in row['text']:
            text = common.cleanhtml(common.remove_username(common.remove_url(ftfy.fix_text(row['text']))))
            hastags = re.findall(r"#(\w+)", text)
            if len(hastags) != 0:
                for tag in hastags:
                    if tag.lower() not in hashtag:
                        hashtag[tag.lower()] = 1
                    else:
                        hashtag[tag.lower()] += 1
    print(hashtag)
    with open('./intermediate_data/hastags.json', 'w') as outfile:
        json.dump(hashtag, outfile)

def check_hashtag_f(source):
    frequency = []
    with open(source) as json_file:
        hashtags = json.load(json_file)
        for hashtag in hashtags:
            if hashtags[hashtag] not in frequency:
                frequency.append(hashtags[hashtag])
    # print(sorted(frequency, reverse = True))
    with open('./intermediate_data/sorted_hastags.json', 'w') as json_file:
        for w in sorted(hashtags, key = hashtags.get, reverse = True):
            if hashtags[w] >= 10:
                json_file.write(w + ':' + str(hashtags[w]) + '\n')

if __name__ == "__main__":
    #step 1 extract userid and tweets
    # unique_userid_list = extract_username('./intermediate_data/hpv_geotagged.csv')
    # with open('./intermediate_data/userid_list.json', 'w') as outfile:
    #    json.dump(unique_userid_list, outfile)

    # step 2 calculate numebr of unique user id and max and min tweets per user
    # 99227 unique uid ; Max: 7341; Min:1; user with only one tweets: 72188; user more than one tweets: 27039;
    # calculate_tweets_per_user('./intermediate_data/userid_list.json')

    # step 3 extract hastag
    extract_hashtag('./intermediate_data/hpv_geotagged.csv')
    check_hashtag_f('./intermediate_data/hastags.json')