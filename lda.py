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
import random_generate_tweets

import operator

import pprint
pp = pprint.PrettyPrinter(indent=4)


import nltk
from nltk.stem import WordNetLemmatizer


import gensim
from gensim import corpora, models, similarities

import numpy as np
import matplotlib.pyplot as plt

import random

def load_stoplist():
    stoplist = set()
    with open('./english.stop.txt', newline='', encoding='utf-8') as f:
        for line in f:
            stoplist.add(line.strip())
    return stoplist

def filter_by_frequency(texts, cnt = 3):
    # remove words that appear only once
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    return [[token for token in text if frequency[token] > 1] for text in texts]

# pattern = r'''(?x)    # set flag to allow verbose regexps
#     [A-Z]\.+\S+        # abbreviations, e.g. U.S.A.
#     | \w+-\w+\S+        # words with optional internal hyphens
# '''
    # | \$?\d+(\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
    # | \.\.\.            # ellipsis
    # | [][.,;"'?():-_`]  # these are separate tokens; includes ], [
# >>> nltk.regexp_tokenize(text, pattern)

pattern=r'[A-Z]\.+\S+|\w+\-\w+|\w+'

def load_data(csv_file, text_fields = []):

    data = pd.read_csv(csv_file, encoding='utf-8')
    
    documents = []
    for i, r in data.iterrows():

        document = ''
        for text_field in text_fields:
            #logger.info(pd.isnull(r[text_field]))
            if(pd.notnull(r[text_field])):
                document = '%s  %s'%(document, common.cleanhtml(common.remove_hashtag_sign(common.remove_username(common.remove_url(ftfy.fix_text(r[text_field]))))))
                # document = '%s  %s'%(document, r[text_field])

        documents.append(document)

    logger.info("# of documents: %d"%len(documents))

    stoplist = load_stoplist()
    # logging.info(stoplist)
    # quit()
    wordnet_lemmatizer = WordNetLemmatizer()

    texts = [[wordnet_lemmatizer.lemmatize(word.lower()) for word in nltk.regexp_tokenize(document, pattern) if wordnet_lemmatizer.lemmatize(word.lower()) not in stoplist] for document in documents]


    # # quit()

    texts = filter_by_frequency(texts)

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    # logger.info(corpus[0])
    return dictionary, corpus


def train_num_of_topics(dictionary, corpus, ranges, output_figure):

    #random.shuffle(corpus)

    test_l = int(len(corpus) * 0.2)
    train_corpus = corpus[test_l+1:]
    test_corpus = corpus[:test_l]

    stats = {
        'perplex': [],
        'per_word_perplex': [],
        'log_perplexity': []
    }
    num_topics_list = ranges #range(5, 151, 5)

    number_of_words = sum(cnt for document in test_corpus for _, cnt in document)
    for num_topics in num_topics_list:

        lda = gensim.models.ldamodel.LdaModel(corpus=train_corpus, id2word=dictionary, num_topics=num_topics, eval_every=10, alpha='auto', chunksize=10000, passes=20, iterations = 500)

        perplex = lda.bound(test_corpus)
        #logger.info(perplex)
        
        per_word_perplex = np.exp2(-perplex / number_of_words) 
        log_perplexity = lda.log_perplexity(test_corpus)

        stats['perplex'].append(perplex)
        stats['per_word_perplex'].append(per_word_perplex)
        stats['log_perplexity'].append(log_perplexity)

    #logger.info(stats)
    # for num_topics in num_topics_list:
    #     logger.info('[%d]: perplex: %.3f; per_world_perlex: %.3f'%(num_topics, stats['perplex'][num_topics], stats['per_word_perplex']))

    ax = plt.figure(figsize=(7, 4), dpi=600).add_subplot(111)
    plt.plot(num_topics_list, stats['per_word_perplex'], color="#254F09")
    plt.xlim(num_topics_list[0], num_topics_list[-1])
    plt.ylabel('Per Word Perplexity')
    plt.xlabel('topics')
    plt.title('')
    plt.savefig(output_figure, format='png', bbox_inches='tight', pad_inches=0.1)
    #plt.show()

def lda_train(dictionary, corpus, num_topics):
    number_of_words = sum(cnt for document in corpus for _, cnt in document)

    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, eval_every=10, alpha='auto', chunksize=10000, passes=20, iterations = 500)

    lda.print_topics(num_topics=num_topics)
    return lda

def lda_infer(lda_model_file, csv_file, dictionary):
    pass
    

def highlight_color_func(word, highlight_words, font_path, font_size, position, orientation, random_state=None):
    if random_state is None:
        random_state = random.Random()

    if (word in highlight_words):
        return "hsl(0, 100%, 50%)"
    else:
        return "hsl(0, 0%%, %d%%)" % random.randint(10, 40)


def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(10, 40)


def wordcloud(topics=[]):
    from wordcloud.wordcloud import WordCloud

    for label, freqs in topics:
        # logger.info(label)
        # logger.info(freqs[0])
        # quit()
        highlight_words = [];
        wordcloud = WordCloud(color_func = grey_color_func, random_state=1, margin=10, background_color='white').fit_words(freqs)
        # wordcloud.to_file("./all_data/figures/adv_in_nj/hpv.%s.tagcloud.png"%(label))
        # wordcloud.to_file("./intermediate_data/promotional/25tp/hpv.%s.tagcloud.png"%(label))
        # wordcloud.to_file("./intermediate_data/laypeople/15tp/hpv.%s.tagcloud.png"%(label))
        wordcloud.to_file("./intermediate_data/hpv_tweets/35tp/hpv.%s.tagcloud.png"%(label))
def fix_utf8(csv_file):

    with open(csv_file, 'rb') as rf, open('%s.fixed.utf-8'%csv_file, 'w', encoding='utf-8') as wf:
        f = rf.read().decode('utf-8', 'ignore')
        wf.write(ftfy.fix_text(f))

fieldnames = ['id', 'text', 'clean_text', 'place', 'user_location', 'us_state', 'created_at', 'username', 'user_id','class','is_quote_status','topic']
def to_csv(tweets, csv_output_file):
        with open(csv_output_file, 'w', newline='', encoding='utf-8') as csv_f:#, open(txt_output_file, 'w', newline='', encoding='utf-8') as txt_f:
            writer = csv.DictWriter(csv_f, fieldnames=fieldnames, delimiter=',', quoting=csv.QUOTE_ALL)
            writer.writeheader()
            for tweet in tweets:

                writer.writerow(tweet)

fieldnames_annotate_themes = ['topic_id', 'tweets','probability']
def to_csv_annotate_template(topic_result, prob, csv_output_file):
        with open(csv_output_file, 'w', newline='', encoding='utf-8') as csv_f:#, open(txt_output_file, 'w', newline='', encoding='utf-8') as txt_f:
            writer = csv.DictWriter(csv_f, fieldnames=fieldnames_annotate_themes, delimiter=',', quoting=csv.QUOTE_ALL)
            writer.writeheader()
            for tid in topic_result:
                prob_ranked = sorted(prob[tid],reverse=True)
                for i in range(10):
                    writer.writerow({
                        'topic_id' : tid,
                        'tweets' : topic_result[tid][prob_ranked[i]],
                        'probability': prob_ranked[i]
                    })


def topic_annotation(lda_model, dictionary, path, csv_output_file):
    tweets = []
    with open(path, 'r', newline='', encoding='utf-8') as csv_f:
        reader = csv.DictReader(csv_f)
        # prob_result = []
        topic_result = {}
        cnt = 0
        prob = {}
        test = []
        for row in reader:
            stoplist = load_stoplist()
            wordnet_lemmatizer = WordNetLemmatizer()
            text = [wordnet_lemmatizer.lemmatize(word.lower()) for word in nltk.regexp_tokenize(row['clean_text'], pattern) if wordnet_lemmatizer.lemmatize(word.lower()) not in stoplist]
            doc_bow = dictionary.doc2bow(text)
            doc_lda = lda_model[doc_bow]
            row['topic'] = []
            # logger.info(doc_lda)
            cnt += 1
            # logger.info(cnt)
            for tp in doc_lda:
                if tp[1] > 0.3:
                    if tp[0] not in topic_result:
                        topic_result[tp[0]] = {float(tp[1]) : row['clean_text']}
                        prob[tp[0]] = [float(tp[1])]
                        test.append(row['clean_text'])
                    else:
                        if row['clean_text'] not in test:
                            prob[tp[0]].append(float(tp[1]))
                            topic_result[tp[0]][float(tp[1])] = row['clean_text']
                            test.append(row['clean_text'])
        for tid in topic_result:
            if tid ==1 or tid == 10 or tid == 13 or tid == 12:
                logger.info(tid)
                logger.info(topic_result[tid])
        # to_csv_annotate_template(topic_result, prob, csv_output_file)
        # logger.info(cnt)

def topic_distribution(lda_model, dictionary, path, csv_output_file):
    tweets = []
    with open(path, 'r', newline='', encoding='utf-8') as csv_f:
        reader = csv.DictReader(csv_f)
        # prob_result = []
        cnt = 0
        total = 0
        for row in reader:
            text = [word for word in row['clean_text']]
            doc_bow = dictionary.doc2bow(text)
            doc_lda = lda_model[doc_bow]
            row['topic'] = []
            total += 1
            for tp in doc_lda:
                if tp[1] >= 0.15 :
                    row['topic'].append(tp[0])
            if len(row['topic']) != 0 :
                cnt += 1
            tweets.append(row)
        to_csv(tweets, csv_output_file)
        logger.info(total)
        logger.info(cnt)

def infer_pz_d(k,lda_model,dictionary,input_folder,output_folder):
    for i in range(k):
        pz_d = []
        with open(input_folder + str(i) + 'tp.csv', 'r', newline='', encoding='utf-8') as csv_f:
            reader = csv.DictReader(csv_f)
            for row in reader:
                stoplist = load_stoplist()
                wordnet_lemmatizer = WordNetLemmatizer()
                text = [wordnet_lemmatizer.lemmatize(word.lower()) for word in nltk.regexp_tokenize(row['clean_text'], pattern) if wordnet_lemmatizer.lemmatize(word.lower()) not in stoplist]
                doc_bow = dictionary.doc2bow(text)
                doc_lda = lda_model[doc_bow]
                topic_distribution = [0] * k
                for tp in doc_lda:
                    topic_distribution[tp[0]] = tp[1]
                pz_d.append(topic_distribution)
        with open(output_folder + str(i) + 'tp.txt', "w") as text_file:
            for line in pz_d:
                result = ''
                for p in line:
                    result += str(p) + ' '
                text_file.write(result)
                text_file.write('\n')


def plot_topics(x,c_y,d_y,a_y):
    plt.plot(x,c_y)
    plt.plot(x,d_y)
    plt.plot(x,a_y)
    plt.xlabel('#topics')
    plt.legend(['CaoJuan2009', 'Deveaud2014', 'Arun2010'], loc='upper right')
    plt.savefig('./intermediate_data/promotional/#_of_topics', format='png', bbox_inches='tight', pad_inches=0.1)
    # plt.show()

def find_topics(dictionary, corpus, topics):
    a_y = []
    d_y = []
    c_y = []
    for num_topic in topics:
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topic, eval_every=10, alpha='auto', chunksize=10000, passes=5)
        a_y.append(Arun2010.FindTopicsNumber(dictionary, corpus, num_topic, model))
        d_y.append(Deveaud2014.FindTopicsNumber(dictionary, corpus, num_topic, model))
        c_y.append(CaoJuan2009.FindTopicsNumber(dictionary, corpus, num_topic, model))
    c_y = [(y / max(c_y) * 10) for y in c_y]
    d_y = [(y / max(d_y) * 10) for y in d_y]
    c_y = [(y / max(d_y) * 10) for y in c_y]
    logger.info(a_y)
    logger.info(d_y)
    logger.info(c_y)
    plot_topics(topics,c_y,d_y,a_y)

topic_stats_filenames = ['topic_id','1st','2nd','3rd','4th','5th','6th','7th','8th','9th','10th']
def to_csv_topic_stats(topics, path):
    with open(path, 'w', newline='', encoding='utf-8') as csv_f:
            writer = csv.DictWriter(csv_f, fieldnames=topic_stats_filenames, delimiter=',', quoting=csv.QUOTE_ALL)

            writer.writeheader()
            for topic_tuple in topics:
                writer.writerow({'topic_id': topic_tuple[0],
                                '1st': topic_tuple[1][0][0] + '(' + str(round(topic_tuple[1][0][1],4)) + ')',
                                '2nd': topic_tuple[1][1][0] + '(' + str(round(topic_tuple[1][1][1],4)) + ')',
                                '3rd': topic_tuple[1][2][0] + '(' + str(round(topic_tuple[1][2][1],4)) + ')',
                                '4th': topic_tuple[1][3][0] + '(' + str(round(topic_tuple[1][3][1],4)) + ')',
                                '5th': topic_tuple[1][4][0] + '(' + str(round(topic_tuple[1][4][1],4)) + ')',
                                '6th': topic_tuple[1][5][0] + '(' + str(round(topic_tuple[1][5][1],4)) + ')',
                                '7th': topic_tuple[1][6][0] + '(' + str(round(topic_tuple[1][6][1],4)) + ')',
                                '8th': topic_tuple[1][7][0] + '(' + str(round(topic_tuple[1][7][1],4)) + ')',
                                '9th': topic_tuple[1][8][0] + '(' + str(round(topic_tuple[1][8][1],4)) + ')',
                                '10th': topic_tuple[1][9][0] + '(' + str(round(topic_tuple[1][9][1],4)) + ')',
                                })


if __name__ == "__main__":

    logger.info(sys.version)


    # twitter data
    # BASE = './intermediate_data/laypeople/laypeople'
    # BASE = './intermediate_data/hpv_tweets/hpv_tweets'
    # BASE = './intermediate_data/promotional/promotional'
    BASE = './intermediate_data/LDA_BTM_comparison/LDA/lda_BTM_comparison_traning_data'
    INPUT = '%s.csv'%BASE
    TEXT_FIELDS = ['clean_text']
    SOURCE = 'twitter'


    DICT = '%s.hpv.dict'%BASE
    MM = '%s.hpv.mm'%BASE

    # fix_utf8(INPUT)
    # quit()
    # Step 1: Create dictionary and corpus
    dictionary, corpus = load_data(INPUT, TEXT_FIELDS)
    dictionary.save(DICT)
    corpora.MmCorpus.serialize(MM, corpus)
    # quit()


    # Step 2:
    # dictionary = corpora.Dictionary.load(DICT)
    # corpus = corpora.MmCorpus(MM)
    # find_topics(dictionary, corpus, range(15, 80, 4))
    # train_num_of_topics(dictionary, corpus, range(10, 40, 1), '%s.train_num_of_topics_by_perplexity.png'%INPUT)
    # quit()

    # Step 3: train lda
    dictionary = corpora.Dictionary.load(DICT)
    corpus = corpora.MmCorpus(MM)

    num_topics = 11
    # lda = lda_train(dictionary, corpus, num_topics = num_topics)
    MODEL = '%s.%d.lda.model'%(BASE,num_topics)
    # lda.save(MODEL)

    lda = gensim.models.ldamodel.LdaModel.load(MODEL)

    # wordcloud(lda.show_topics(num_topics=num_topics, formatted=False), './figures/%s'%(SOURCE))
    # wordcloud(topics=lda.show_topics(num_topics=num_topics, formatted=False))
    # quit()
    # topics=lda.show_topics(num_topics=num_topics, num_words=10, formatted=False)
    # logger.info(topics)
    # to_csv_topic_stats(topics,'./intermediate_data/analysis/laypeople_topics_keywords_stats_15.csv')

    # step 4: topic_annotation_LDA
    # topic_annotation(lda,dictionary,'./intermediate_data/laypeople/laypeople.csv','./intermediate_data/laypeople/15tp/annotation.csv', )

    #step 5: topic_distribution_LDA
    # topic_distribution(lda,dictionary,'./intermediate_data/promotional/promotional.csv', './intermediate_data/promotional/cutoffline/0.085.csv')

    #step 6: infer p(z|d) for each doc
    infer_pz_d(num_topics,lda,dictionary,'./intermediate_data/LDA_BTM_comparison/sample_cluster_csv/','./intermediate_data/LDA_BTM_comparison/LDA/topics_distribution_cluster/')