# Load the raw training data and replace any missing values with NA
training.data.raw <- read.csv('SpamProcessedData.csv', header=T, na.strings=c(""))

str(training.data.raw)
data <- training.data.raw
data$label
#need spam and ham to be a factor
data$label <- factor(data$label)
is.factor(data$label)

#process data
library(tm)
library(SnowballC)
library(Matrix)
library(ggplot2)
library(ROCR)
library(caret)
library(dplyr)
library(e1071)

dtm <- Corpus(VectorSource(data$SMSMessage)) %>%
  tm_map(removeNumbers) %>%
  tm_map(stripWhitespace) %>%
  tm_map(removeWords, stopwords()) %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removePunctuation) %>%
  tm_map(stemDocument) %>%
  DocumentTermMatrix()

index = sample(5559, 5559*0.8)

train_matrix <- as.matrix(dtm[index,])
test_matrix <- as.matrix(dtm[-index,])

dtm_train <- Matrix(train_matrix, sparse=T)
dtm_test <- Matrix(test_matrix, sparse=T)

train_labels <- data[index, ]$label
test_labels <- data[-index, ]$label

prop.table(table(train_labels))

fit_ridge <- glmnet(dtm_train, train_labels, family = 'binomial', alpha = 0)
fit_lasso <- glmnet(dtm_train, train_labels, family = 'binomial', alpha = 1)

pred <- predict(fit_ridge, dtm_test, type = 'response')
head(pred[,1:11])
 
head(coef(fit_ridge, s=1.52))

glmnet_pred <- predict(fit_ridge, newx = dtm_test, s = c(1.52, 0.1))
head(glmnet_pred)

#plot(fit_ridge, xlab = 'Ridge Method')
#abline(0,0)

#plot(fit_lasso, xlab = 'Lasso Method')
#abline(0,0)

#comparison
glmnet_fit <- cv.glmnet(x=dtm_train, y = train_labels, family = 'binomial', alpha = 0)
head(coef(glmnet_fit, s = 'lambda.min'))
plot(glmnet_fit)
glmnet_fit$lambda.min

glmnet_lasso <- cv.glmnet(x = dtm_train, y = train_labels, family = 'binomial', alph = 1)
plot(glmnet_lasso)
glmnet_lasso$lambda.min

preds_logit <- predict(glmnet_fit, newx = dtm_test, type='response', s = 'lambda.min')
head(preds_logit)

summary(preds_logit)

preds_newlogit <- rep('0', length(preds_logit))
preds_newlogit[preds_logit >= 0.5] <- '1'

#print confusion matrix
confusionMatrix(test_labels, preds_newlogit)

#create data frame
results <- data.frame(pred=preds_logit, actual = test_labels)
ggplot(results, aes(x = preds_logit, fill = actual)) + geom_density(alpha = 0.2)

prediction_logit <- prediction(preds_logit, test_labels)
perf <- performance(prediction_logit, measure = "tpr", x.measure = "fpr")

#area under the curve
auc <- performance(prediction_logit, measure = "auc")
auc <- auc@y.values[[1]]
auc

#ROC curve
roc_logit <- data.frame(fpr = unlist(perf@x.values), tpr = unlist(perf@y.values))

ggplot(roc_logit, aes(x = fpr, ymin = 0, ymax = tpr)) + 
  geom_ribbon(alpha = 0.1) +
  geom_line(aes(y = tpr)) +
  geom_abline(slope = 1, intercept = 0, linetype = 'dashed') +
  ggtitle("ROC Curve") +
  ylab('True Positive Rate') +
  xlab('False Positive Rate')
