# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 21:44:33 2018
sources: https://towardsdatascience.com/spam-classifier-in-python-from-scratch-27a98ddd8e73
@author: Notorious-V
"""

import nltk as lang
#use this if you do not have nltk module
#lang.download('all')
import numpy as np
import matplotlib.pyplot as plot
#from wordcloud import WordCloud
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
'''def createWordCloud(data):
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
    plot.show()'''

# measure the length of the message
def messageLength(data):
    messageLen = len(data)
    return messageLen

#get upper case word count
def checkUpper(data):
    count = 0
    for index, value in enumerate(data):
        if value.isupper():
            count += 1
    return count

#check if message has a website url
def hasWebsite(data):
    #print(data)
    for index, value in enumerate(data):
        if(value.startswith('www') or value.startswith('http//') or value.startswith('ftp//')):
            return 1
    return 0

#check for the most frequent words
def mostFrequentWords(data, word=True):   
    counts = Counter(data)
    
    if(word):
        item = counts.most_common(1)
        for index, value in enumerate(item):
            item = value[1]    
        return item
    else:
        #return string literal word
        item = counts.most_common(1)
        for index, value in enumerate(item):
            item = value[0]
        return item
    
def preProcessMessage(data,stop_words = True, stemm = True, lower = True, grams = 2, tokenize = True, punctuation = True):
    #set the words to lowercase, this helps us to not deal with free or FREE as two different words 
    #print(data)
    if(lower):
        data = data.str.lower()
        #remove punctuations
    if(punctuation):
        data = data.str.translate(str.maketrans("","",'|-[]_.:;,!?&()''""\\'))
        print(data)
        #tokenize the data
        
    if(tokenize):
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
            
    if tokenize:
        return tokens
    return data

def main():
    '''trainfeatureFrame = []
    testfeatureFrame = []'''
    featureFrame = []
    #set column width to display data
    pd.set_option('display.max_colwidth', 100)
    #read file and create the dataframe
    data = pd.read_csv('./dataset/SMSSpamCollection.txt', sep='\t', quoting=csv.QUOTE_NONE, names=["label", "SMS Message"], encoding="utf8")
    #data = data.replace({"spam": 1, "ham": 0})
    df = pd.DataFrame(data)
    
    #split the data into test and training set
    '''trainingSet, testSet = list(), list()
    for i in range(df['SMS Message'].shape[0]):
        if np.random.uniform(0, 1) < 0.50:
            trainingSet += [i]
        else:
            testSet += [i]
    train = df.loc[trainingSet]
    test = df.loc[testSet]
    
    train.reset_index(inplace=True)
    train.drop(['index'], axis = 1, inplace=True)
    
    test.reset_index(inplace=True)
    test.drop(['index'], axis = 1, inplace=True)'''
    
    #wordcloud = WordCloud().generate_from_frequencies(train['SMS Message'])
    #print(wordcloud)
    
    #process messages before createing a word cloud for optimization
    #train['SMS Message'] = preProcessMessage(train['SMS Message'])
    
    # we do not have to split the data until tested in r
    #df['SMS Message'] = preProcessMessage(df['SMS Message'])
    
    #test['SMS Message'] = preProcessMessage(test['SMS Message'])
    
    #need to tokenize before searching a url and convert to lowercase
    df['SMS Message'] = preProcessMessage(df['SMS Message'], True, False, True, 2, True, True)
    
    for i in range(len(df)):
      featureFrame.append(hasWebsite(df['SMS Message'][i]))
     
    #append feature to the data frame
    df2 = pd.DataFrame(featureFrame)
    df['Website'] = df2
    
    featureFrame = []
    
    #get most frequent word count
    for i in range(len(df)):
        featureFrame.append(mostFrequentWords(df['SMS Message'][i]))
    
    #append feature to the data frame
    df2 = pd.DataFrame(featureFrame)
    df['F-WordCount'] = df2
    
    featureFrame = []
    
    #get most frequent word
    for i in range(len(df)):
        featureFrame.append(mostFrequentWords(df['SMS Message'][i], False))
    print(featureFrame)
    
    df2 = pd.DataFrame(featureFrame)
    df['F-word'] = df2
    
    print(df)
    
    # prep message for most frequent word count and extraction, lowercase, tokenizing, stemming, punctuation and n-gram is completed.
    #df['SMS Message'] = preProcessMessage(df['SMS Message'], True, False, False, 1, False, False)
    
    #print(df['Website'])
    #preProcessMessage(data,stop_words = True, stemm = True, lower = True, grams = 2, tokenize = True, punctuation = True):
    #df['SMS Message'] = preProcessMessage(df['SMS Message'], True, False, True, 2, True, True)
    #print(df['SMS Message'])
    # frequent word count
    '''for i in range(len(df)):
        featureFrame.append(mostFrequentWords(df['SMS Message'][i]))
    print(featureFrame)
    #append feature to the data frame
    df2 = pd.DataFrame(trainfeatureFrame)
    train['F3'] = df2
    
    # frequent word
    for i in range(len(df)):
        featureFrame.append(mostFrequentWords(df['SMS Message'][i], False))
    print(featureFrame)
    
    df2 = pd.DataFrame(testfeatureFrame)
    test['F3'] = df2
    
    for i in range(len(train)):
      trainfeatureFrame.append(messageLength(train['SMS Message'][i]))
      
    for i in range(len(test)):
      testfeatureFrame.append(messageLength(test['SMS Message'][i]))
      
    #append feature to the data frame
    df2 = pd.DataFrame(trainfeatureFrame)
    train['F2'] = df2
    
    df2 = pd.DataFrame(testfeatureFrame)
    test['F2'] = df2
    #print(train)
    trainfeatureFrame = []
    testfeatureFrame = []
    
    
    
    #remove the sms message column
    train = train.drop(['SMS Message'], axis = 1)
    test = test.drop(['SMS Message'], axis = 1)
    
    #print(test)
    #print(train)
    #print(train.isna())
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
    train.to_csv('./trainingData.csv', encoding='utf-8-sig')
    test.to_csv('./testData.csv', encoding='utf-8-sig')'''
    df.to_csv('./SpamProcessedData.csv', encoding='utf-8-sig')
    print('done')
main()
