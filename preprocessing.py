#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import logging
import logging.handlers

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

import os
import re
import json
import sys
import time
import csv
import common
import langid
from geocoding.tweet_us_state_geocoder import TweetUSStateGeocoder

tug = TweetUSStateGeocoder()

def filter_raw(raw_json_data_folder):
    tweets = []
    # texts = set()
    tweetIds = set()
    cnt_total_no_duplicate =0
    geocoded_cnt = 0
    lang_cnt = 0
    place_cnt = 0
    gps_cnt = 0
    duplicated_cnt = 0
    sanitized_cnt = 0
    total_cnt = 0
    for root, dirs, files in os.walk(os.path.abspath(raw_json_data_folder)):
        for f in files:
            if (f != 'search.json' and not f.startswith('.')):
                logger.info(os.path.join(root, f))

                with open(os.path.join(root, f), 'r') as json_file:

                    for line in json_file:

                        try:
                            if (line.startswith('{')):
                                tweet = json.loads(line)
                                total_cnt += 1
                                if (int(tweet['id']) in tweetIds):
                                    duplicated_cnt += 1
                                    continue
                                else:
                                    tweetIds.add(int(tweet['id']))
                                cnt_total_no_duplicate += 1

                                if ('lang' in tweet and 'en' != tweet['lang']):
                                    continue

                                if ('lang' not in tweet):
                                    lang, prob = langid.classify(text)
                                    if (lang != 'en'):
                                        continue
                                lang_cnt += 1

                                sanitized_text = common.sanitize_text(tweet['text']).strip()
                                if sanitized_text == '' or sanitized_text == 'RT':
                                    continue
                                sanitized_cnt += 1

                                geolocation = tweet['user']['location']

                                if 'place' in tweet and tweet['place']:

                                    if (tweet['place']['country_code'] != 'US'):
                                        continue
                                    else:
                                        geolocation = tweet['place']['full_name']

                                us_state = tug.get_state(geolocation)

                                if (not us_state):
                                    continue

                                geocoded_cnt += 1

                                tweets.append({
                                    'id': tweet['id'],
                                    'text':tweet['text'],
                                    'clean_text': sanitized_text,
                                    'place': tweet['place'] if 'place' in tweet else '',
                                    'user_location': tweet['user']['location'],
                                    'us_state': us_state,
                                    'created_at': tweet['created_at'],
                                    'username': tweet['user']['name'],
                                    'user_id': tweet['user']['id'],
                                    'is_quote_status': tweet['is_quote_status']
                                    })
                        except Exception as exc:
                            logger.info(line)
                            logger.warn('ignore: %s'%(exc))

    logger.info('total: %d; duplicate: %d; no_duplicate: %d; en: %d; sanitize_text: %d; geo: %d'%(total_cnt, duplicated_cnt, cnt_total_no_duplicate, lang_cnt, sanitized_cnt, geocoded_cnt))

    return tweets

fieldnames = ['id', 'text', 'clean_text', 'place', 'user_location', 'us_state', 'created_at', 'username', 'user_id','is_quote_status']
# fieldnames = ['id', 'text', 'clean_text', 'place', 'user_location', 'created_at', 'username', 'user_id']
def to_csv(tweets, csv_output_file):
    #tweets = []
    with open(csv_output_file, 'w', newline='', encoding='utf-8') as csv_f:#, open(txt_output_file, 'w', newline='', encoding='utf-8') as txt_f:
        
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames, delimiter=',', quoting=csv.QUOTE_ALL)

        writer.writeheader()
        
        for tweet in tweets:

            writer.writerow(tweet)


if __name__ == "__main__":
    
    logger.info(sys.version)


    tweets = filter_raw('./raw_data')
    to_csv(tweets, './intermediate_data/hpv_geotagged.csv')
