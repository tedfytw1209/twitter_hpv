#!/usr/bin/env python
#coding=utf-8


import logging
import logging.handlers

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

import sys
import os
import time
import common
import csv
import operator

def count_word_freq(source, outputfile):
	word_freq = {}
	with open(source, 'r', encoding = 'utf-8') as input_file:
		all_tweets = input_file.readlines()
		for tweet in all_tweets:
			wv = tweet.strip().split(' ')
			for word in wv:
				if word not in word_freq:
					word_freq[word] = 1
				else:
					word_freq[word] += 1
	sorted_x = sorted(word_freq.items(), key=operator.itemgetter(1), reverse = True)
	with open(outputfile, 'w', encoding = 'utf-8') as json_file:
		for w in sorted_x:
			json_file.write(w[0] + ':' + str(w[1]) + '\n')

if __name__ == '__main__':
	count_word_freq('./intermediate_data/hpv_tweets/hpv_tweets.txt', './intermediate_data/hpv_tweets/word_freq.json')