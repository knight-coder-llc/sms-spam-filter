# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 21:44:33 2018
sources: https://towardsdatascience.com/spam-classifier-in-python-from-scratch-27a98ddd8e73
@author: Notorious-V
"""

import nltk as lang
#lang.download('all')
import numpy as np
import matplotlib.pyplot as plot
from wordcloud import WordCloud
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from collections import Counter
import pandas as pd
#import sklearn as sk
import csv
import os
import re
import string

#create spam word cloud
def createWordCloud(data):
    s_words = ' '.join(list(data[data['label'] == 1]['SMS Message']))
    word_count = WordCloud(width = 512, height = 512).generate(s_words)
    plot.figure(figsize = (10, 8), facecolor = 'k')
    plot.imshow(word_count)
    plot.axis('off')
    plot.tight_layout(pad = 0)
    print('SPAM')
    plot.show()
    print('\n')
    #create ham word cloud
    h_words = ' '.join(list(data[data['label'] == 0]['SMS Message']))
    word_count = WordCloud(width = 512, height = 512).generate(h_words)
    plot.figure(figsize = (10, 8), facecolor = 'k')
    plot.imshow(word_count)
    plot.axis('off')
    plot.tight_layout(pad = 0)
    print('HAM')
    plot.show()

# measure the length of the message
def messageLength(data):
    for i in range(data):
        messageLen = len(data[i])
        return messageLen

#check if message has a website url
def hasWebsite(data):
    #print(data)
    for index, value in enumerate(data):
        if(value.startswith('www') or value.startswith('http//') or value.startswith('ftp//')):
            return True
    return False

#check for the most frequent words
def mostFrequentWords(data, frame):   
    counts = Counter(data)
    item = counts.most_common(1)
    #print(data)
    
    print(item)
    
def preProcessMessage(data,stop_words = True, stemm = True, lower = True, grams = 2):
    #set the words to lowercase, this helps us to not deal with free or FREE as two different words 
    if(lower):
        data = data.str.lower()
        #remove punctuations
        data = data.str.translate(str.maketrans("","",'|-[]_.:;,!?&()''""\\'))
        #tokenize the data
        tokens = [lang.word_tokenize(token) for token in data]

    #use n-grams to improve accuracy
    if(grams > 1):
        token = []
        for i in range(len(tokens)):
            for j in range(len(tokens[i])):
                token += [' '.join(tokens[i][j])]
    #remove the stopwords to improve efficiency
    if(stop_words):
        stop = stopwords.words('english')
        for index, value in enumerate(tokens):
            tokens[index] = [token for token in value if token not in stop]
        
    #the snowball stemmer is better than porter stemming per documentation, removes redundancy between word meanings.
    if(stemm):
        stemmer = SnowballStemmer('english')
        for index, value in enumerate(tokens):
            tokens[index] = [stemmer.stem(token) for token in value]
    return tokens

def main():
    featureFrame = []
    #set column width to display data
    pd.set_option('display.max_colwidth', 100)
    #read file and create the dataframe
    data = pd.read_csv('./dataset/SMSSpamCollection.txt', sep='\t', quoting=csv.QUOTE_NONE, names=["label", "SMS Message"], encoding="utf8")
    data = data.replace({"spam": 1, "ham": 0})
    df = pd.DataFrame(data)
    
    #split the data into test and training set
    trainingSet, testSet = list(), list()
    for i in range(df['SMS Message'].shape[0]):
        if np.random.uniform(0, 1) < 0.50:
            trainingSet += [i]
        else:
            testSet += [i]
    train = df.loc[trainingSet]
    test = df.loc[testSet]
    
    train.reset_index(inplace=True)
    train.drop(['index'], axis = 1, inplace=True)
    
    
    #wordcloud = WordCloud().generate_from_frequencies(train['SMS Message'])
    #print(wordcloud)
    
    #process messages before createing a word cloud for optimization
    
    train['SMS Message'] = preProcessMessage(train['SMS Message'])
    
    for i in range(len(train)):
      featureFrame.append(hasWebsite(train['SMS Message'][i]))
      
    #append feature to the data frame
    df2 = pd.DataFrame(featureFrame)
    train['F1'] = df2
    print(train)
    print(train['SMS Message'])
    
    #call feature 2
    
    
    #print(len(featureFrame))
    
    #words = mostFrequentWords(train['SMS Message'])
    #mostFrequentWords(train['SMS Message'])
    #isweb = hasWebsite(train['SMS Message'][9], featureFrame)
    #print(isweb)
    
    #print(words)
    #print(train['SMS Message'])
    #print(train['SMS Message'])
    #most frequent word count graphic for spam and ham messages
    #createWordCloud(data)
    
    #create and export the processed dataset?
    #train.to_csv('./trainingData.csv', encoding='utf-8-sig')
    #test.to_csv('./testData.csv', encoding='utf-8-sig')
    print('done')
main()
