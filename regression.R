# Load the raw training data and replace any missing values with NA
training.data.raw <- read.csv('SpamProcessedData.csv', header=T, na.strings=c(""))

# lets output the number of any missing values
sapply(training.data.raw, function(x) sum(is.na(x)))

# check how many different values for each feature
sapply(training.data.raw, function(x) length(unique(x)))

# visualize missing data
library(Amelia)
missmap(training.data.raw, main = "Missing values vs observed")

#subset the data
data <- training.data.raw
data$label
# substitute missing data with average
#data$F.WordCount[is.na(data$F.WordCount)] <- mean(data$F.WordCount, na.rm=T)

# check factors
#is.factor(data$label)
#is.factor(data$Website)
#is.factor(data$W.count)
#is.factor(data$F.WordCount)
#is.factor(data$Spamword)

# check categorical variable encoding to understand the model
#factor(data$label)
#contrasts(data$label)
#contrasts(data$F.WordCount)

# remove rows in F3 with NAs
#data <- data[!is.na(data$F.WordCount),]
#rownames(data) <- NULL

#split the data for training and testing set
train <- data[1: 2501,]
test <- data[2502:5002,]

# Model fitting
model <- glm(train$label ~ train$Spamword + train$Website + train$W.count + train$F.WordCount, family=binomial(), data=train)
#model
#measure predictability ### 
#####################################
new <- data.frame(testset = test$Spamword + test$Website + test$W.count + test$F.WordCount)
fitted.results <- predict(model, newdata=new, type='response')

fitted.results <- ifelse(fitted.results > 0.5, "1", "0")
fitted.results
fitted.results.errors <- mean(fitted.results != test$label)
fitted.results.accuracy <- 1 - fitted.results.errors
print(paste('Model Accuracy =', round(fitted.results.accuracy*100, 2), "%"))

library(ROCR)
library(caret)
# ROC and AUC
p <- predict(model, test ,type='response')
#summary(p)
#predict(model, type="response")

acc <- ifelse(p > .5, 0, 1)
confusionMatrix(acc, test$label, positive="1") 

pr <- prediction(p, test$label)
summary(pr)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)

auc <- performance(pr, measure = "auc")
