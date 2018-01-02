#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import logging
import logging.handlers

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')


import os
import sys
import math
import csv
import common
import random

fieldnames = ['id', 'text', 'clean_text', 'place', 'user_location', 'us_state', 'created_at', 'username', 'user_id','class','is_quote_status']
fieldnames = ['clean_text', 'us_state','preprocessed_text']
def to_csv(tweets, csv_output_file):
        with open(csv_output_file, 'w', newline='', encoding='utf-8') as csv_f:#, open(txt_output_file, 'w', newline='', encoding='utf-8') as txt_f:
            writer = csv.DictWriter(csv_f, fieldnames=fieldnames, delimiter=',', quoting=csv.QUOTE_ALL)
            writer.writeheader()
            for tweet in tweets:
                writer.writerow({
                    'clean_text': tweet['clean_text'],
                    'us_state': tweet['us_state'],
                    'preprocessed_text': tweet['preprocessed_text']})

# fieldnames = ['clean_text']
# def to_csv(tweets, csv_output_file):
#         with open(csv_output_file, 'w', newline='', encoding='utf-8') as csv_f:#, open(txt_output_file, 'w', newline='', encoding='utf-8') as txt_f:
#             writer = csv.DictWriter(csv_f, fieldnames=fieldnames, delimiter=',', quoting=csv.QUOTE_ALL)
#             writer.writeheader()
#             for tweet in tweets:
#                 writer.writerow({
#                     'clean_text': tweet})

def random_generate_tweets_cvs(input_file,output_file):
    samples_number = random.sample(range(1, 271533), 100)
    logger.info(samples_number)
    sample_data = []
    result = []
    with open(input_file, 'r', newline='', encoding='utf-8') as csv_f:
        reader = csv.DictReader(csv_f)
        for row in reader:
            sample_data.append(row)

    for i in samples_number:
        result.append(sample_data[i])

    to_csv(result,output_file)

def random_generate_tweets_txt(k):
    for i in range(k):
        with open('./intermediate_data/LDA_BTM_comparison/clusters/' + str(i) + 'tp.txt', encoding='utf-8') as f:
            lines = f.readlines()
            if len(lines) >= 1000:
                samples_number = random.sample(range(1, len(lines)), 1000)
                result = []
                for j in range(len(lines)):
                    if j in samples_number:
                        # print(j)
                        result.append(lines[j])
                with open('./intermediate_data/LDA_BTM_comparison/sample_cluster_txt/' + str(i) + 'tp.txt', "w", encoding='utf-8') as text_file:
                    for line in result:
                        text_file.write(line)
                to_csv(result,'./intermediate_data/LDA_BTM_comparison/sample_cluster_csv/' + str(i) + 'tp.csv')
            else:
                with open('./intermediate_data/LDA_BTM_comparison/sample_cluster_txt/' + str(i) + 'tp.txt', "w", encoding='utf-8') as text_file:
                    for line in lines:
                        text_file.write(line)
                to_csv(lines,'./intermediate_data/LDA_BTM_comparison/sample_cluster_csv/' + str(i) + 'tp.csv')

def random_generate_pz_d(input_folder, output_folder, k):
    for i in range(k):
        with open(input_folder + 'tp' + str(i) + '.pz_d') as f:
            lines = f.readlines()
            if len(lines) > 800:
                samples_number = random.sample(range(1, len(lines)), 800)
                result = []
                for j in range(len(lines)):
                    if j in samples_number:
                        # print(j)
                        result.append(lines[j])
                with open(output_folder + 'tp' + str(i) + '.pz_d', "w") as text_file:
                    for line in result:
                        text_file.write(line)
            else:
                with open(output_folder + 'tp' + str(i) + '.pz_d', "w") as text_file:
                    for line in lines:
                        text_file.write(line)

if __name__ == "__main__":

    logger.info(sys.version)

    #random generate tweets
    # random_generate_tweets_cvs('./intermediate_data/preprocessed_text_and_geo.csv','./intermediate_data/analysis/BTM/cutoffline_annotation/random_100.csv')


    # random generate tweets for BTM and LDA comparision
    # k = 11
    # random_generate_tweets_txt(k)


    # random generate pz_d for BTM
    for k in range(2,3):
    # for k in [5,10,15,20,25,30]:
        random_generate_pz_d('./intermediate_data/BTM/nostopwords/tp' + str(k) + '_clusters/','./sample_cluster_stopwords/tp' + str(k) + '_clusters/',k)