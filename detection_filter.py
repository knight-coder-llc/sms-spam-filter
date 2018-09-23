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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import sklearn as sk  
import pandas as pd
import csv
import arff

#create spam word cloud
def createWordCloud(data):
    s_words = ' '.join(list(data[data['label'] == 1]['SMSMessage']))
    word_count = WordCloud(width = 512, height = 512).generate(s_words)
    plot.figure(figsize = (10, 8), facecolor = 'k')
    plot.imshow(word_count)
    plot.axis('off')
    plot.tight_layout(pad = 0)
    print('SPAM')
    plot.show()
    print('\n')
    #create ham word cloud
    h_words = ' '.join(list(data[data['label'] == 0]['SMSMessage']))
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
        if(value.startswith('www') or value.startswith('http//') or value.startswith('https//') or value.startswith('ftp//')):
            return 1
    return 0

#check for the most frequent words
def mostFrequentWords(data, word=True):   
    counts = Counter(data)
    if(word):
        item = counts.most_common(1)
        for index, value in enumerate(item):
            item = value[1]
            if item >= 3:
                return 1
        return 0
    else:
        #return string literal word
        item = counts.most_common(1)
        for index, value in enumerate(item):
            item = value[0]
        return item



def wordCount(data):
    length = len(data)
    
    if length > 10:
        return 1
    return 0
 

#spam word checker
def spamWords(data, spam):
    
    #spamWordList = ['free', 'urgent', 'call', 'freemsg', 'mob', 'txt', 'entry','reply','claim','download', '2']
    
    if any(map(lambda each: each in data, spamWordList)) == True:
        return 1
    else:
        return 0

def preProcessMessage(data,stop_words = True, stemm = True, lower = True, grams = 2, tokenize = True, punctuation = True):
    
    #set the words to lowercase, lists and series is tricky 
    if(lower):
        data = data.str.lower()
    
    #use n-grams to improve accuracy
    if(grams > 1):
        '''token = []
        for i in range(len(tokens)):
            for j in range(len(tokens[i])):
                token += [' '.join(tokens[i][j])]
        #print(tokens)'''
        
        
        '''vectorizer = TfidfVectorizer(any(data),ngram_range=(1,2),encoding="utf-8", lowercase=False, strip_accents="unicode", stop_words="english", norm= 'l1')
        vectorizer.fit_transform(data)
        
        spamlist = vectorizer.get_feature_names()'''
        
    #remove punctuations
    if(punctuation):
        data = data.str.translate(str.maketrans("","",'<>|-[]_.:;,!?&()''""\\'))
         
    
        
    #tokenize the data    
    if(tokenize):
        tokens = [lang.word_tokenize(token) for token in data]

    
        #print(token)
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


#testing thinking about implementing naive bayes theorm
def create_dictionary(data):
    '''data = '' + data.apply(lambda x: ' '.join(x))
    
    vectorizer = TfidfVectorizer(any(data),ngram_range=(1,2),encoding="utf-8", lowercase=False, strip_accents="unicode", stop_words="english", norm= 'l2')
    #
    train = vectorizer.fit_transform(data)
    train_x, test_x, train_y, test_y = train_test_split(train[1], train[0])
    classifier = LogisticRegression()
    classifier.fit(train_x, train)
    
    print(vectorizer.vocabulary_)
    '''#counts = Counter(data)
    
    '''items = []
    for i in range(len(data)):
        counts = Counter(data[i])    
        item = counts.most_common(1)
        items.append(item)'''

    #print(counts.most_common(3000))

def featureExtract(df, action=None):
    featureFrame = []
    for i in range(len(df)):
      featureFrame.append(action(df['SMSMessage'][i]))
      
    return featureFrame
    
trainPositive = {}
trainNegative = {}



def naive_bayes(data):
    pA = float(0)
    pNa = float(0)
    
    total = 0
    numSpam = 0
    for email in data['label']:
        if email == 1:
            numSpam += 1
        total += 1
    
    pA = numSpam / float(total) 
    pNa = (total - numSpam)/float(total)
    
    return pA, pNa

 

    
def processEmail(data, body, label):
    positivetotal = 0
    negativetotal = 0
    
    tokens = [lang.word_tokenize(token) for token in data['SMSMessage']]
    itemsPos = []
    itemsNeg = []
    #get total ham and total spam, needs to be tokenized
    for i in range(len(tokens)):
        for index, word in enumerate(tokens[i]):
           
            if data['label'][i] == 1:
                trainPositive[word] = trainPositive.get(word, 0) + 1
                #seperate the data
                itemsPos.append(tokens[i])
                positivetotal += 1   
            else:
                trainNegative[word] = trainNegative.get(word, 0) + 1
                #seperate the data
                itemsNeg.append(tokens[i])
                negativetotal += 1
    
    pA, pNa = naive_bayes(data)
    return trainPositive
    #classify(itemsPos, itemsNeg, pA, pNa, positivetotal, negativetotal, trainPositive,trainNegative)

#need to figure out what this is doing with tp word
def conditionalWord(word, total, tp_n):
    return tp_n[word]/float(total) 
        
#not understanding how I am getting a non type error
def conditionalEmail(body, total, tp_n):
    result = 1.0
    for word in body:
          result += conditionalWord(word, total, tp_n)
    return result

def classify(emailPos, emailNeg, pA, pNa, pT, nT, tp, tn):

    #tokens passed, bug somewhere here in the passing of tp and tn
    for i in range(len(emailPos)):
        isSpam = pA * conditionalEmail(emailPos[i], pT, tp)
        print(isSpam)
    for i in range(len(emailNeg)):
        notSpam = pNa * conditionalEmail(emailNeg[i], nT, tn)
        #print(notSpam)
        '''if isSpam > notSpam:
            result.append(1)
        else:
            result.append(0)'''
    #print(isSpam)
    #print(notSpam)

def main():
    
    #set column width to display data
    pd.set_option('display.max_colwidth', 100)
    #read file and create the dataframe
    data = pd.read_csv('./dataset/SMSSpamCollection.txt', sep='\t', quoting=csv.QUOTE_NONE, names=["label", "SMSMessage"], encoding="utf8")
    data = data.replace({"spam": 1, "ham": 0})
    df = pd.DataFrame(data)
    
    isSpam = processEmail(df,df['SMSMessage'][1], df['label'][1]) 
    #print(isSpam)
    #need to tokenize before searching a url and convert to lowercase
    df['SMSMessage'] = preProcessMessage(df['SMSMessage'])
    
    df['Website'] = featureExtract(df,hasWebsite)
    df['W-count'] = featureExtract(df,wordCount)
    df['F-WordCount'] = featureExtract(df,mostFrequentWords)
    df['Spamword'] = featureExtract(df,spamWords, isSpam)
    
    #create and export the processed dataset?
    df.to_csv('./SpamProcessedData.csv', encoding='utf-8-sig')
    
    #translate the message data back to string values for arff.dump
    df['SMSMessage'] = '' + df['SMSMessage'].apply(lambda x: ' '.join(x))
    #df.drop(['SMSMessage'], axis=1)
    arff.dump('spam.arff',df.values , relation="spam", names=df.columns)
    #print(arff.dumps('spam.arff',df.values, relation="spam"))'''
    
    print('done')
main()
