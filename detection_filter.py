# -*- coding: utf-8 -*-
"""
Created by: Brian Kilburn
Date: 9/15/2018
Purpose: SMS Spam Filter
"""

import nltk as lang
#use this if you do not have nltk module
#lang.download('all')
import numpy as np
import matplotlib.pyplot as plot
from wordcloud import WordCloud
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from collections import Counter
import pandas as pd
import csv

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

def wordCount(data):
    return len(data)
    
#spam word checker
def spamWords(data):
    #print(data)
    spamWordList = ['free', 'urgent', 'call', 'freemsg', 'mob', 'txt', 'entry','reply','claim','download', '2']
    
    if any(map(lambda each: each in data, spamWordList)) == True:
        return 1
    else:
        return 0
    
def preProcessMessage(data,stop_words = True, stemm = False, lower = True, grams = 1, tokenize = True, punctuation = True):
    
    # convert to string if needed
    '''if(toString):
        data = data.apply(lambda x: ' '.join(x))'''
    
    #set the words to lowercase, lists and series is tricky 
    if(lower):
        data = data.str.lower()
    
    #remove punctuations
    if(punctuation):
        data = data.str.translate(str.maketrans("","",'<>|-[]_.:;,!?&()''""\\'))
         
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
    
    featureFrame = []
    #set column width to display data
    pd.set_option('display.max_colwidth', 100)
    #read file and create the dataframe
    data = pd.read_csv('./dataset/SMSSpamCollection.txt', sep='\t', quoting=csv.QUOTE_NONE, names=["label", "SMS Message"], encoding="utf8")
    #data = data.replace({"spam": 1, "ham": 0})
    df = pd.DataFrame(data)
    
    
    #need to tokenize before searching a url and convert to lowercase
    df['SMS Message'] = preProcessMessage(df['SMS Message'])
    
    for i in range(len(df)):
      featureFrame.append(hasWebsite(df['SMS Message'][i]))
    
    #append feature to the data frame
    df2 = pd.DataFrame(featureFrame)
    df['Website'] = df2
    
    featureFrame = []
    
    #word count
    for i in range(len(df)):
        featureFrame.append(wordCount(df['SMS Message'][i]))
    
    df2 = pd.DataFrame(featureFrame)
    df['W-count'] = df2
    
    featureFrame = []
    
    #get most frequent word count, data is a list
    for i in range(len(df)):
        featureFrame.append(mostFrequentWords(df['SMS Message'][i]))
    
    #append feature to the data frame
    df2 = pd.DataFrame(featureFrame)
    df['F-WordCount'] = df2
    
    featureFrame = []
    
    #spam word detector
    for i in range(len(df)):
        featureFrame.append(spamWords(df['SMS Message'][i]))
    
    df2 = pd.DataFrame(featureFrame)
    df['Spamword'] = df2
    
    featureFrame = []
    
    #get most frequent word
    for i in range(len(df)):
        featureFrame.append(mostFrequentWords(df['SMS Message'][i], False))
    
    df2 = pd.DataFrame(featureFrame)
    df['F-word'] = df2
    
    featureFrame = []
    
    #create and export the processed dataset?
    df.to_csv('./SpamProcessedData.csv', encoding='utf-8-sig')
    print('done')
main()
