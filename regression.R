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
data <- subset(training.data.raw, select=c(2,4,5,6,7))

# substitute missing data with average
data$F.WordCount[is.na(data$F.WordCount)] <- mean(data$F.WordCount, na.rm=T)

# check factors
is.factor(data$label)
is.factor(data$Website)
is.factor(data$W.count)
is.factor(data$F.WordCount)
is.factor(data$Spamword)

# check categorical variable encoding to understand the model
contrasts(data$label)
#contrasts(data$F.WordCount)

# remove rows in F3 with NAs
data <- data[!is.na(data$F.WordCount),]
rownames(data) <- NULL

#split the data for training and testing set
train <- data[1: 2001,]
test <- data[2002:4002,]
test
# Model fitting
model <- glm(label ~ train$Website + train$Spamword, family=binomial(link='logit'), data=train)
model
#measure predictability ### 
#####################################
fitted.results <- predict(model, newdata=subset(test,select=test$Website + test$Spamword), type='response')
fitted.results
fitted.results <- ifelse(fitted.results > 0.5, 1, 0)
fitted.results
library(ROCR)
# ROC and AUC
p <- predict(model, newdata=test,type='response')
pr <- prediction(p, test$label)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)

auc <- performance(pr, measure = "auc")
auc