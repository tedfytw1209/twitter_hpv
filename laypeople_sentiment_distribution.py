#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import logging
import logging.handlers

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')


import os
import json
import sys
import csv
import pandas as pd
from datetime import datetime
import pytz

filename = ['us_state','positive','negative','neutral']
def to_csv(results,csv_output_file):
    with open(csv_output_file, 'a', newline='', encoding='utf-8') as csv_f:
            writer = csv.DictWriter(csv_f, fieldnames=filename, delimiter=',', quoting=csv.QUOTE_ALL)
            writer.writeheader()
            for state in results:
                    writer.writerow({
                        'us_state': state,
                        'positive': results[state][1],
                        'negative': results[state][-1],
                        'neutral': results[state][0]})

def laypeople_sentiment_distibution(preditction_f, original_f, output_f):
    dates = []
    df = pd.read_csv(original_f, encoding = "utf-8")
    for index, row in df.iterrows():
            date = datetime.strptime(row['created_at'],'%a %b %d %H:%M:%S +0000 %Y').replace(tzinfo=pytz.UTC)
            y_m = str(date.year) + '-' + (str(date.month) if (len(str(date.month)) == 2) else ('0'+str(date.month)))
            dates.append(y_m)

    result = {}
    df = pd.read_csv(preditction_f, encoding = "ISO-8859-1")
    for index, row in df.iterrows():
        if '2016' in dates[index]:
            if row['us_state'] not in result:
                result[row['us_state']] = {-1: 0, 1: 0, 0: 0}
                result[row['us_state']][row['label']] = 1
            else:
                if row['label'] not in result[row['us_state']]:
                    result[row['us_state']][row['label']] = 1
                else:
                    result[row['us_state']][row['label']] += 1
    to_csv(result,output_f)


if __name__ == "__main__":

    logger.info(sys.version)
    laypeople_sentiment_distibution('./intermediate_data/laypeople/laypeople.predicted.csv','./intermediate_data/laypeople/laypeople.csv', './intermediate_data/laypeople/laypeople_sentiment_distibution_2016.csv')