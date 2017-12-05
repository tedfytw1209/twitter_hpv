#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import logging
import logging.handlers

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

import csv
import os
import json
import pandas as pd
from datetime import datetime
import pytz
import matplotlib.pyplot as plt

def filter_raw(raw_json_data_folder):
    tweets = []

    # texts = set()
    tweetIds = set()
    cnt_total_no_duplicate =0
    duplicated_cnt=0
    cnt = 0
    for root, dirs, files in os.walk(os.path.abspath(raw_json_data_folder)):
        for f in files:
            if (f != 'search.json' and not f.startswith('.')):
                logger.info(os.path.join(root, f))

                with open(os.path.join(root, f), 'r') as json_file:

                    for line in json_file:

                        try:
                            if (line.startswith('{')):
                                tweet = json.loads(line)
                                if (int(tweet['id']) in tweetIds):
                                    duplicated_cnt += 1
                                    continue
                                cnt_total_no_duplicate += 1
                                tweets.append({
                                    'id': tweet['id'],
                                    'created_at': tweet['created_at']
                                    })

                                tweetIds.add(int(tweet['id']))

                        except Exception as exc:
                            logger.info(line)
                            logger.warn('ignore: %s'%(exc))

    logger.info('duplicate: %d; no_duplicate: %d'%(duplicated_cnt,cnt_total_no_duplicate))
    return tweets

# fieldnames = ['id', 'created_at']
fieldnames = ['year/month','tweets']
def to_csv(tweets, csv_output_file):
    #tweets = []
    with open(csv_output_file, 'w', newline='', encoding='utf-8') as csv_f:

        writer = csv.DictWriter(csv_f, fieldnames=fieldnames, delimiter=',', quoting=csv.QUOTE_ALL)

        writer.writeheader()

        for tweet in tweets:

            writer.writerow(tweet)

def count_tweets_by_month(source):
    df = pd.read_csv(source)
    trend = {}
    result = []
    for index, row in df.iterrows():
        date = datetime.strptime(row['created_at'],'%a %b %d %H:%M:%S +0000 %Y').replace(tzinfo=pytz.UTC)
        y_m = str(date.year) + '-' + (str(date.month) if (len(str(date.month)) == 2) else ('0'+str(date.month)))
        if y_m not in trend:
            trend[y_m] = 1
        else:
            trend[y_m] += 1
    logger.info(trend)
    for date in trend:
        result.append({
            'year/month' : date,
            'tweets' : trend[date]
        })
    return result

def plot_trend(source):
    fields = ['year-month','tweets_new']
    df = pd.read_csv(source,usecols=fields,encoding='utf-8')
    names = df['year-month']
    data = df['tweets_new'].dropna()
    logger.info(data)
    ax = plt.subplot(111)
    width=0.5
    bins = list(map(lambda x: x-width/2,range(1,len(data)+1)))
    ax.bar(bins,data,width=width,color='k',align='edge')
    ax.set_xticks(list(map(lambda x: x, range(1,len(data)+1))))
    ax.set_xticklabels(names,rotation=0)
    plt.ylabel('number of tweets')
    plt.xlabel('year/month')
    plt.show()

if __name__ == "__main__":

    #step 1: generate no duplicate tweets csv file
    # tweets = filter_raw('./raw_data')
    # to_csv(tweets, './intermediate_data/raw_tweets_time.csv')

    #step 2: count tweets by month
    # result = count_tweets_by_month('./intermediate_data/raw_tweets_time.csv')
    # to_csv(result,'./intermediate_data/month_volume.csv')

    #step 3: plot
    plot_trend('./intermediate_data/month_volume.csv')