#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import logging.handlers

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s]: %(levelname)s: %(message)s')

import os
import sys
import time
import pandas as pd
import csv

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import pearsonr

def correlation(input_f):

    positive_negative = []
    for root, dirs, files in os.walk(os.path.abspath(input_f)):
        for f in files:
            logger.info(f)
            df = pd.read_csv(os.path.join(root, f),encoding='utf-8')
            positive = np.array([])
            negative = np.array([])
            for index, row in df.iterrows():
                positive = np.append(positive, np.asarray(row['5th'] + row['7th']))
                negative = np.append(negative, np.asarray(row['1st'] + row['2nd'] + row['4th']))
                positive_negative = np.divide(positive, negative)
            logger.info(positive)
            logger.info(negative)
            logger.info(positive_negative)
            quit()


if __name__ == "__main__":
    correlation('./intermediate_data/analysis/BTM/tweets_distribution/state_level/')