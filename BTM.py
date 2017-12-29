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
import common
import csv
import ftfy
# from sortedcontainers import SortedDict

import pprint
pp = pprint.PrettyPrinter(indent=4)
import nltk
from nltk.stem import WordNetLemmatizer

import numpy as np
# import matplotlib.pyplot as plt

import random


# return:    {wid:w, ...}
def read_voca(pt):
    voca = {}
    for l in open(pt, encoding='utf-8'):
        wid, w = l.strip().split('\t')[:2]
        voca[int(wid)] = w
    return voca

def read_pz(pt):
    return [float(p) for p in open(pt,encoding='utf-8').readline().split()]

# voca = {wid:w,...}
def dispTopics(pt, voca, pz):
    k = 0
    topics = []
    for l in open(pt):
        vs = [float(v) for v in l.split()]
        wvs = zip(range(len(vs)), vs)   # (for a specific topic, wvs store (wordid, propability) )
        wvs = sorted(wvs, key=lambda d:d[1], reverse=True)

        # tmps = ' '.join(['%s:%f' % (voca[w],v) for w,v in wvs[:10]])
        tmps = [(voca[w],v) for w,v in wvs[:15]]
        topics.append((k, tmps))
        k += 1
    return topics

def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(10, 40)


def wordcloud(topics,k):
    from wordcloud.wordcloud import WordCloud

    for label, freqs in topics:
        highlight_words = [];
        wordcloud = WordCloud(color_func = grey_color_func, random_state=1, margin=10, background_color='white').fit_words(freqs)

        wordcloud.to_file("./intermediate_data/figures/BTM_wordcould/" + str(k) + "tp/hpv.%s.tagcloud.png"%(label))


def generate_corpus_for_quality_evaluation(k,pz_d,tweets,topic_words_distribution):
    all_tweets = []
    logger.info(k)
    df = pd.read_csv(tweets,encoding='utf-8')
    for index, row in df.iterrows():
            all_tweets.append(row['tweets'])

    tweets_pz_d = []
    with open(pz_d) as f:
        for l in f:
            line = l.strip().split(' ')
            tweets_pz_d.append([float(p) for p in line])

    results = {}
    for j in range(len(tweets_pz_d)):
        if 'nan' not in tweets_pz_d[j] and '-nan' not in tweets_pz_d[j]:
            sorted_pz_ds = list(tweets_pz_d[j])
            sorted_pz_ds.sort(reverse = True)
            topic_id = tweets_pz_d[j].index(sorted_pz_ds[0])
            if topic_id not in results:
                results[topic_id] = [all_tweets[j]]
            else:
                results[topic_id].append(all_tweets[j])

    final_result = []
    for tp in results:
        for keyword in topic_words_distribution[tp][1]:
            temp = []
            dedup = set()
            for tweet in results[tp]:
                if '%s'%keyword[0] in tweet.lower():
                    clean_text_list = (common.cleanhtml(common.remove_username(common.remove_url(ftfy.fix_text(tweet.lower()))))).strip(' ').replace('\n', ' ').split(' ')[:-1]
                    clean_text = ",".join(str(x) for x in clean_text_list)
                    if clean_text not in dedup:
                        temp.append(tweet)
                        dedup.add(clean_text)

            # samples_number = random.sample(range(1, len(temp)), 1)
            # if (tp == 6) and (keyword[0] == 'u.s.'):
            #     logger.info(temp)
            #     quit()

            samples_number = []            
            if len(temp) <= 2:
                samples_number = range(len(temp))
            else:
                samples_number = random.sample(range(1, len(temp)), 2)
            for i in samples_number:
                result = {}
                result['topic_id'] = tp
                result['keyword'] = keyword[0]
                result['propability'] = keyword[1]
                result['tweet'] = temp[i]
                final_result.append(result)

    to_csv(final_result, '../../papers/2017_BMC_HPV/analysis/BTM/quality_evaluation/'+str(k) + 'tp.csv')

fieldnames = ['topic_id', 'keyword', 'propability','tweets']
def to_csv(results, csv_output_file):
        with open(csv_output_file, 'a', newline='', encoding='utf-8') as csv_f:
            writer = csv.DictWriter(csv_f, fieldnames=fieldnames, delimiter=',', quoting=csv.QUOTE_ALL)
            writer.writeheader()
            for tweet in results:
                    writer.writerow({
                        'topic_id': tweet['topic_id'],
                        'keyword': tweet['keyword'],
                        'propability': tweet['propability'],
                        'tweets': tweet['tweet']
                        })


def transfer_to_word_id(input_f, output_f, k):
    voca = {}
    with open('./Biterm/output/' + str(k) + 'tp/voca.txt', 'r') as vc:
        for l in vc:
            wid, w = l.strip().split('\t')[:2]
            voca[w] = int(wid)

    tweets = []
    with open(input_f, 'r',encoding='utf-8') as clusters:
        for line in clusters:
            ws = line.strip().split()
            tweets.append(ws)
    with open(output_f,'w') as w_id:
        for tweet in tweets:
            for w in tweet:
                if w in voca:
                    w_id.write(str(voca[w]) + ' ')
            w_id.write('\n')

def cutoffline_stats(input_f, n):
    pz_d = []
    with open(input_f, 'r') as f:
        for line in f:
            pz_d.append(line.strip().split(' '))

    cnt = 0
    for p_topics in pz_d:
        for p_topic in p_topics:
            if float(p_topic) >= n:
                cnt += 1
                break

    logger.info(cnt)

def generate_cutoffline_file(pz_d,csv_f,output_f, cutoffline):
    result = []
    tweet_topics_distribution = []
    with open(pz_d, 'r') as pz_d_f:
        for l in pz_d_f:
            topic_id = []
            line = l.strip().split(' ')
            for i in range(len(line)):
                if float(line[i]) >= cutoffline:
                    topic_id.append(i)
            tweet_topics_distribution.append(topic_id)

    df = pd.read_csv(csv_f,encoding='utf-8')
    for index, row in df.iterrows():
        temp ={}
        temp['clean_text'] = row['clean_text']
        temp['topic_id'] = tweet_topics_distribution[index]
        result.append(temp)
    to_csv(result, output_f+str(cutoffline)+'.csv')

# filenames_topics = ['us_state','1st','2nd','3rd','4th','5th','6th','7th']
states = ["ny", "mo", "pa", "nj", "ri", "wa", "ca", "il", "ky", "tx", "fl", "dc", "co", "ct", "va", "oh", "in", "ma", "mi", "ok", "me", "ms", "tn", "ga", "nc", "ia", "ut", "mn", "md", "wi", "ne", "sc", "sd", "or", "ks", "mt", "az", "nv", "nh", "nd", "hi", "al", "ar", "vt", "nm", "la", "ak", "de", "id", "wv", "wy"]
months = ['2017-10','2017-09','2017-08','2017-07','2017-06','2017-05','2017-04','2017-03','2017-02','2017-01','2016-12','2016-11','2016-10','2016-09','2016-08','2016-07','2016-06','2016-05','2016-04','2016-03','2016-02','2016-01','2015-12','2015-11','2015-10','2015-09','2015-08','2015-07','2015-06','2015-05','2015-04','2015-03','2015-02','2015-01','2014-12','2014-11','2014-10','2014-09','2014-08','2014-07','2014-06','2014-05','2014-04','2014-03','2014-02','2014-01']
filenames_topics = ['y_m','1st','2nd','3rd','4th','5th','6th','7th']
def to_csv_tweets_distribution(results,csv_output_file):
    logger.info(csv_output_file)
    with open(csv_output_file, 'a', newline='', encoding='utf-8') as csv_f:
            writer = csv.DictWriter(csv_f, fieldnames=filenames_topics, delimiter=',', quoting=csv.QUOTE_ALL)
            writer.writeheader()
            # for state in results:
            #      writer.writerow({
            #                 'y_m': state,
            #                 '1st': results[state][0],
            #                 '2nd': results[state][1],
            #                 '3rd': results[state][2],
            #                 '4th': results[state][3],
            #                 '5th': results[state][4],
            #                 '6th': results[state][5],
            #                 '7th': results[state][6]})
            for month in months:
                    logger.info(month)
                    if month not in results:
                        writer.writerow({
                        'y_m': month,
                        '1st': 0,
                        '2nd': 0,
                        '3rd': 0,
                        '4th': 0,
                        '5th': 0,
                        '6th': 0,
                        '7th': 0})
                    else:
                        writer.writerow({
                            'y_m': month,
                            '1st': results[month][0],
                            '2nd': results[month][1],
                            '3rd': results[month][2],
                            '4th': results[month][3],
                            '5th': results[month][4],
                            '6th': results[month][5],
                            '7th': results[month][6]})

def tweets_distribution(input_f, pz_d, cutoffline):
    tweet_topics_distribution = []
    with open(pz_d, 'r') as pz_d_f:
        for l in pz_d_f:
            topic_id = []
            line = l.strip().split(' ')
            for i in range(len(line)):
                if float(line[i]) >= cutoffline:
                    topic_id.append(i)
            tweet_topics_distribution.append(topic_id)

    result = {}
    df = pd.read_csv(input_f,encoding='utf-8')
    for index, row in df.iterrows():
        if '2016' in row['date']:
            if row['us_state'] not in result:
                result[row['us_state']] = [0] * 7
                for tid in tweet_topics_distribution[index]:
                    result[row['us_state']][tid] = 1
            else:
                for tid in tweet_topics_distribution[index]:
                    result[row['us_state']][tid] += 1

    to_csv_tweets_distribution(result,'./intermediate_data/analysis/BTM/tweets_distribution/distribution_2016.csv')

def tweets_distribution_by_month(input_f, pz_d, cutoffline):
    tweet_topics_distribution = []
    with open(pz_d, 'r') as pz_d_f:
        for l in pz_d_f:
            topic_id = []
            line = l.strip().split(' ')
            for i in range(len(line)):
                if float(line[i]) >= cutoffline:
                    topic_id.append(i)
            tweet_topics_distribution.append(topic_id)

    result = {}
    df = pd.read_csv(input_f,encoding='utf-8')
    for index, row in df.iterrows():
        if row['date'] not in result:
            result[row['date']] = [0] * 7
            for tid in tweet_topics_distribution[index]:
                result[row['date']][tid] = 1
        else:
            for tid in tweet_topics_distribution[index]:
                result[row['date']][tid] += 1
    to_csv_tweets_distribution(result,'./intermediate_data/analysis/BTM/tweets_distribution/distribution_by_month.csv')



def tweets_distribution_by_month_by_state(input_f, pz_d, cutoffline, state):
    tweet_topics_distribution = []
    with open(pz_d, 'r') as pz_d_f:
        for l in pz_d_f:
            topic_id = []
            line = l.strip().split(' ')
            for i in range(len(line)):
                if float(line[i]) >= cutoffline:
                    topic_id.append(i)
            tweet_topics_distribution.append(topic_id)

    result = {}
    df = pd.read_csv(input_f,encoding='utf-8')
    for index, row in df.iterrows():
        if row['us_state'] == state:
            if row['date'] not in result:
                result[row['date']] = [0] * 7
                for tid in tweet_topics_distribution[index]:
                    result[row['date']][tid] = 1
            else:
                for tid in tweet_topics_distribution[index]:
                    result[row['date']][tid] += 1

    to_csv_tweets_distribution(result,'./intermediate_data/analysis/BTM/tweets_distribution/state_level/'+state+'.csv')


def count_tweet_by_state(source):
    state_count = {}
    df = pd.read_csv(source,encoding='utf-8')
    for index, row in df.iterrows():
        if row['us_state'] not in state_count:
            state_count[row['us_state']] = 1
        else:
            state_count[row['us_state']] += 1

    with open('./intermediate_data/state_count.csv', 'a', newline='', encoding='utf-8') as csv_f:
            writer = csv.DictWriter(csv_f, fieldnames=['state','number'], delimiter=',', quoting=csv.QUOTE_ALL)
            writer.writeheader()
            for state in state_count:
                 writer.writerow({
                            'state': state,
                            'number': state_count[state]})

if __name__ == "__main__":

    logger.info(sys.version)

    for K in range(11,12):

        model_dir = 'Biterm/output/%dtp/model/' % K
        voca_pt = 'Biterm/output/%dtp/voca.txt' % K
        voca = read_voca(voca_pt)

        pz_pt = model_dir + 'k%d.pz' % K
        pz = read_pz(pz_pt)

        zw_pt = model_dir + 'k%d.pw_z' %  K
        # topics = dispTopics(zw_pt, voca, pz)

        # tweets = './intermediate_data/hpv_tweets/hpv_tweets_not_by_uid.csv'
        tweets = './intermediate_data/hpv_tweets/hpv_tweets.txt'
        pz_d = model_dir + 'k%d.pz_d' % K

        # get wordcould figures
        # wordcloud(topics, K)

        # generate corpus for evaluating quality of K
        # generate_corpus_for_quality_evaluation(K,pz_d,tweets,topics)

    # transfer tweets txt to word id file
    # transfer_to_word_id('./intermediate_data/analysis/BTM/cutoffline_annotation/random_100.txt', './intermediate_data/analysis/BTM/cutoffline_annotation/100_doc_wids.txt', 7)
    # transfer_to_word_id('./intermediate_data/hpv_tweets/hpv_tweets_not_by_uid_BTM_input.txt','./Biterm/output/'+ str(K) +'tp/word_id_not_by_uid.txt', K)

    # count tweets by state
    # count_tweet_by_state('./intermediate_data/hpv_geotagged.csv')

    # test different cutoffline to count tweets
    # n = sys.argv[1]
    # cutoffline_stats('./intermediate_data/k7.pz_d', float(n))

    # generate annotation csv for different cutoffline
    # cutoffline = float(sys.argv[1])
    # generate_cutoffline_file('intermediate_data/analysis/BTM/cutoffline_annotation/100_k7.pz_d','intermediate_data/analysis/BTM/cutoffline_annotation/random_100.csv','intermediate_data/analysis/BTM/cutoffline_annotation/annotation/', cutoffline)

    # generate tweets distribution
    # tweets_distribution('./intermediate_data/preprocessed_text_and_geo_date.csv', './intermediate_data/k7.pz_d', 0.3)
    # tweets_distribution_by_month('./intermediate_data/preprocessed_text_and_geo_date.csv', './intermediate_data/k7.pz_d', 0.3)
    # for state in states:
        # tweets_distribution_by_month_by_state('./intermediate_data/preprocessed_text_and_geo_date.csv', './intermediate_data/k7.pz_d', 0.3, state)