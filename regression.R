#load datasets: nrows= 10,
packages <- c('tm', 'SnowballC', 'caTools', 'rpart', 'rpart.plot', 'randomForest', 'ROCR',"RColorBrewer","e1071","gmodels","NLP")
if (length(setdiff(packages, rownames(installed.packages()))) > 0) {
  install.packages(setdiff(packages, rownames(installed.packages())))  
}
library(tm)
library(SnowballC)
library(caTools)
library(rpart)
library(rpart.plot)
library(randomForest)
library(ROCR)
library(NLP)
library(RColorBrewer)
library(e1071)
library(gmodels)

trainingSet <- read.csv('trainingData.csv', header=TRUE, fileEncoding="UTF-8-BOM", strip.white = TRUE, stringsAsFactors = FALSE, check.names = FALSE)
df <- data.frame(trainingSet)
df <- trainingSet[complete.cases(trainingSet),]

df
sms_corpus <- VCorpus(VectorSource(df))
sms_corpus

sms_corpus_clean <-tm_map(sms_corpus, content_transformer(tolower))
sms_corpus_clean <- tm_map(sms_corpus_clean, removeNumbers)
sms_corpus_clean <-tm_map(sms_corpus_clean, removeWords, stopwords())
sms_corpus_clean <- tm_map(sms_corpus_clean, removePunctuation)
sms_corpus_clean <- tm_map(sms_corpus_clean, stemDocument)
sms_corpus_clean <- tm_map(sms_corpus_clean, stripWhitespace)

sms_dtm <-DocumentTermMatrix(sms_corpus_clean)
sms_data_train <- sms_dtm[1:4169, ]
sms_data_test <- sms_dtm[4170:5559, ]
sms_train_labels <- sms_raw[1:4169, ]$type
sms_test_labels <- sms_raw[4170:5559, ]$type

sms_freq_terms <- findFreqTerms(sms_data_train, 5)
sms_data_freq_train <- sms_data_train[ , sms_freq_terms]
sms_data_freq_test <- sms_data_test[ , sms_freq_terms]
sms_train <- apply(sms_data_freq_train, MARGIN = 2, convert_counts)
sms_test <- apply(sms_data_freq_test, MARGIN = 2, convert_counts)

sms_classifier_train2 <- naiveBayes(sms_train, sms_train_labels, laplace = 1)
sms_test_pred2 <-predict(sms_classifier_train2, sms_test)
CrossTable(sms_test_pred2, sms_test_labels, prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))


#spamlog <- glm(label~., data=df, family="binomial")
#spamlog


