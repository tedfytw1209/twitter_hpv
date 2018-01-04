#!/usr/bin/env python
#coding=utf-8


import logging
import logging.handlers

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

import sys
import numpy as np
import os
import re
import json
import time
import pandas as pd
import ftfy
import common
import csv
import nltk
import math
from scipy import stats
from nltk.stem import WordNetLemmatizer
from multiprocessing import Pool
import concurrent.futures
import functools
import multiprocessing as mp

pattern=r'[A-Z]\.+\S+|\w+\-\w+|\w+'


# def load_stoplist():
#     stoplist = set()
#     with open('./english.stop.txt', newline='', encoding='utf-8') as f:
#         for line in f:
#             stoplist.add(line.strip())
#     return stoplist


def group_tweets_by_cluster_gold_standard(source, k):
    tweets = []
    all_tweets_in_cluster = []
    wordnet_lemmatizer = WordNetLemmatizer()
    
    # read all tweets
    df = pd.read_csv(source,encoding='utf-8')
    for index, row in df.iterrows():
        tweets.append(row['text'])

    hashtags = []
    with open('./intermediate_data/cluster_hashtags.json', 'r') as json_file:
        hashtags = json.load(json_file)
    for i in range(k):
        # with open('./BTM/output/' + str(k) + 'tp/clusters/' + str(i) + 'tp.txt', 'w') as clusters:
        with open('./intermediate_data/LDA_BTM_comparison/clusters/' + str(i) + 'tp.txt', 'w',  encoding="utf-8") as clusters:
            print(i)
            for tweet in tweets:
                tags = re.findall(r"#(\w+)", tweet)
                if len(tags) != 0:
                    for tag in tags:
                        if tag.lower() in hashtags[i]:
                            text = common.cleanhtml(common.remove_hashtag_sign(common.remove_username(common.remove_url(ftfy.fix_text(tweet)))))
                            clean_texts = [wordnet_lemmatizer.lemmatize(word.lower()) for word in nltk.regexp_tokenize(text, pattern)]
                            final_text = ''
                            for word in clean_texts:
                                final_text += word + ' '
                            all_tweets_in_cluster.append(final_text)
                            # final_text = re.sub(r"[\u4e00-\u9fff]", "", final_text)
                            clusters.write(final_text)
                            clusters.write('\n')
                            break

    # txt for BTM
    with open('./intermediate_data/LDA_BTM_comparison/lda_BTM_comparison_traning_data.txt', 'w', encoding="utf-8") as file:
        for tweet in all_tweets_in_cluster:
            file.write(tweet)
            file.write('\n')

    # csv for LDA
    fieldnames = ['clean_text']
    with open('./intermediate_data/LDA_BTM_comparison/lda_BTM_comparison_traning_data.csv', 'w', newline='', encoding='utf-8') as csv_f:
            writer = csv.DictWriter(csv_f, fieldnames=fieldnames, delimiter=',', quoting=csv.QUOTE_ALL)
            writer.writeheader()
            for tweet in all_tweets_in_cluster:
                writer.writerow({'clean_text': tweet})

def generate_tweets_by_cluster_not_gold_standard(source, k):
    tweets_pz_d = []
    with open(source + str(k) + 'tp/model/k' + str(k) + '.pz_d') as topics_distribution:
        for line in topics_distribution:
            pz_ds = []
            temp = line.split(' ')[:k]
            if 'nan' in temp or '-nan' in temp:
                continue
            for pz_d in temp:
                pz_ds.append(float(pz_d))
            tweets_pz_d.append(pz_ds)

    results = {}
    for pz_ds in tweets_pz_d:
        sorted_pz_ds = list(pz_ds)
        sorted_pz_ds.sort(reverse = True)
        for i in range(1):
            topic_id = pz_ds.index(sorted_pz_ds[i])
            if topic_id not in results:
                results[topic_id] = [pz_ds]
            else:
                results[topic_id].append(pz_ds)

    for tp in results:
        with open('./intermediate_data/BTM/nostopwords/tp'+ str(k) +'_clusters/tp' + str(tp) + '.pz_d', 'w') as clusters:
            for pz_ds in results[tp]:
                line = ''
                for pz_d in pz_ds:
                    line += str(pz_d) + ' '
                clusters.write(line)
                clusters.write('\n')

def generate_word_id(k):
    voca = {}
    with open('./Biterm/output/' + str(k) + 'tp/voca.txt', 'r', encoding='utf-8') as vc:
        for l in vc:
            wid, w = l.strip().split('\t')[:2]
            voca[w] = int(wid)

    for i in range(k):
        cluster = []
        print(i)
        with open('./intermediate_data/LDA_BTM_comparison/sample_cluster_txt/' + str(i) + 'tp.txt', 'r', encoding='utf-8') as clusters:
            for line in clusters:
                ws = line.strip().split()
                cluster.append(ws)
            with open('./Biterm/output/' + str(k) + 'tp/word_id_cluster/tp' + str(i)+ '_doc_wids.txt','w', encoding='utf-8') as w_id:
                for tweet in cluster:
                    for w in tweet:
                        if w in voca:
                            w_id.write(str(voca[w]) + ' ')
                    w_id.write('\n')

def dis(di,dj):
    di = np.asarray(di, dtype = np.float)
    dj = np.asarray(dj, dtype = np.float)
    m = 0.5 * (di + dj)
    k_l_l = stats.entropy(di, m)
    k_l_r = stats.entropy(dj, m)  # k_l_1 = np.sum(np.where(di != 0, di * np.log(di / dj), 0))
    dis = 0.5 * (k_l_l + k_l_r)
    return dis

def calculate_h_score_worker(k):
    clusters = []
    for i in range(k):
        # with open('./Biterm/output/' + str(k) + 'tp/topics_distribution_cluster/tp' + str(i)+ '.pz_d','r') as pz_d:
        # with open('./intermediate_data/LDA_BTM_comparison/LDA/topics_distribution_cluster/' + str(i) + 'tp.txt','r') as pz_d:
        with open('./sample_clusters_no_stopwords_v2/tp' + str(k) + '_clusters/tp' + str(i)+ '.pz_d','r') as pz_d:
            cluster = []
            for line in pz_d:
                temp = []
                for p in line.strip().split():
                    temp.append(float(p))
                cluster.append(temp)
        clusters.append(cluster)
    intra_dis = 0
    for t in range(k):
        iteration = 0
        for i in range(len(clusters[t])):
            for j in range(i+1,len(clusters[t])):
                iteration += 2 * dis(clusters[t][i],clusters[t][j]) / (len(clusters[t]) * (len(clusters[t]) - 1))
        intra_dis += (1 / (k + 1)) * iteration

    inter_dis = 0
    for t1 in range(k):
        iteration = 0
        for t2 in range(k):
            if (t2 != t1):
                for i in range(len(clusters[t1])):
                    for j in range(len(clusters[t2])):
                        iteration += dis(clusters[t1][i],clusters[t2][j]) / (len(clusters[t1]) * len(clusters[t2]))
        inter_dis += (1 / ((1 + k) * k)) * iteration
    # h_score = intra_dis / inter_dis
    h_score = np.divide(intra_dis, inter_dis)
    logger.info(k)
    logger.info(h_score)
    return (k,h_score)

def calculate_h_score_worker_callback(future, final_scores = []):
    h_scores = future.result()
    final_scores.append(h_scores)


def calculate_h_score(start, end):
# def calculate_h_score():
    start_time = time.time()
    final_scores = []
    futures_ = []
    # max_workers = mp.cpu_count()
    max_workers = 4
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:

        for k in range(start,end):
        # for k in [20,25,30]:
            future_ = executor.submit(calculate_h_score_worker, k)
            future_.add_done_callback(functools.partial(calculate_h_score_worker_callback, final_scores = final_scores))
            futures_.append(future_)

        else:
            concurrent.futures.wait(futures_)
            executor.shutdown()
            to_csv(final_scores,'./H_score.csv')
            logger.info('processed in [%.2fs]' % ((time.time() - start_time)))

fieldnames = ['h_score', 'k']
def to_csv(h_scores, csv_output_file):
        with open(csv_output_file, 'a', newline='', encoding='utf-8') as csv_f:
            writer = csv.DictWriter(csv_f, fieldnames=fieldnames, delimiter=',', quoting=csv.QUOTE_ALL)
            writer.writeheader()
            for score in h_scores:
                writer.writerow({
                    'h_score': score[1],
                    'k': score[0]})

if __name__ == '__main__':
    start = sys.argv[1]
    end = sys.argv[2]
    # k = 11
    #step 1 generate clusters for gold standard
    # group_tweets_by_cluster_gold_standard('./intermediate_data/hpv_geotagged.csv', k)

    # for k in range(26,30):
    #     generate_tweets_by_cluster_not_gold_standard('./Biterm/output/',k)


    #step 2 generate word_id for BTM
    # generate_word_id(k)

    #step 3 generate pz_d manually for each document BTM


    #step 4 H score single core
    # k = int(k)
    # h_score = calculate_h_score_worker(k)
    # with open('./H_score.txt', 'a') as f:
    #     f.write(str(k) + ':')
    #     f.write(str(h_score))
    #     f.write('\n')

    #step 4 H score multi core
    calculate_h_score(int(start),int(end))
    # calculate_h_score()
