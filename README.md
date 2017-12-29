# Twitter HPV Project

This project is used to analyze HPV data collected from twitter since 2013.


### Prerequisites

* python3
* numpy
* pandas
* gensim

### Data sources

Our data sets are saved on s3 (`s3://ji0ng.bi0n.twitter/hpv/`) collected from 3 data sources.

* `As_adam_201401_201612.zip` from Adam Dunn in Australia.  His data were collected based on the keywords:  `[”Gardasil”, ”Cervarix”, ”hpv + vaccin⁄”, ”cervical + vaccin⁄”]`

* `hpv.random.20151102_20160318.tar.gz` is from UT at Houston based on keywords: `[ 'HPV', 'human papillomavirus', 'Gardasil','Cervarix']`

* The rest are what we have collected based on keywords, `["hpv", "#hpv", "#hpvvaccine", "hpvvaccine", "HumanPapillomavirus", "#HumanPapillomavirus",
   "HumanPapillomavirusVaccine", "#HumanPapillomavirusVaccine",
   "Gardasil", "#Gardasil", "Papillomavirus", "#Papillomavirus", "Cervarix", "#cervarix"
   "cervical cancer", "cervical #cancer", "#cervical cancer", "cervicalcancer", "#cervicalcancer","vph",
  "#vph",
   "#vphvacuna",
   "vphvacuna",
   "viruspapilomahumano",
   "#viruspapilomahumano",
   "viruspapilomahumanovacuna",
   "#viruspapilomahumanovacuna",
   "viruspapiloma",
   "#viruspapiloma",
   "cancer cervical",
   "cancer #cervical",
   "cancercervical",
   "#cancercervical"]
`

### Data analysis

* Processing
    * 1: `count_raw_tweets_by_month.py`: count the number of tweets by month based on the raw tweets (without duplicates).
    * 2: `preporcessing.py` : remove duplicates, remove non-English tweets, clean text (i.e., remove url, hashtags, and usernames), add geo tags, and export as a csv file.
        * total number of tweets: 2507140
        * number of duplicate tweets: 695925
        * number of tweets without duplciates: 1811215
        * number of English tweets: 1277853
        * number of tweets after removing clean_text equals to '' or 'RT': 1274530
        * number of tweets after geo-coding: 271533

* Rule-based categorization
    * `classifier.py`: classify the rest of the tweets into `promotional HPV related` (205,945 tweets) and `laypeople's discussions` (65,588 tweets)

### Topic modeling
* sample size for comparison between LDA and BTM: 15835
    * H_score:
      * BTM: inter: 0.481031355146
      * LDA: inter: 0.781761511916

* lda.py: This code serves two functions:
    * 1: Test different statistical model (Arun2010, CaoJuan2009, Deveaud2014)
    * 2: Train LDA model and visulize the results as word clouds
    * 3: Assign topics to tweets

