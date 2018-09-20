# Load the raw training data and replace any missing values with NA
training.data.raw <- read.csv('trainingData.csv', header=T, na.strings=c(""))

# lets output the number of any missing values
sapply(training.data.raw, function(x) sum(is.na(x)))

# check how many different values for each feature
sapply(training.data.raw, function(x) length(unique(x)))

# visualize missing data
library(Amelia)
missmap(training.data.raw, main = "Missing values vs observed")

#subset the data
data <- subset(training.data.raw, select=c(2,3,4,5))

# substitute missing data with average
data$F2[is.na(data$F2)] <- mean(data$F2, na.rm=T)

# check factors
is.factor(data$label)
is.factor(data$F1)
is.factor(data$F2)
is.factor(data$F3)

# check categorical variable encoding to understand the model
contrasts(data$label)
contrasts(data$F3)

# remove rows in F3 with NAs
data <- data[!is.na(data$F3),]
rownames(data) <- NULL

#process testing data

# Load the raw training data and replace any missing values with NA
test.data.raw <- read.csv('testData.csv', header=T, na.strings=c(""))

# lets output the number of any missing values
sapply(test.data.raw, function(x) sum(is.na(x)))

# check how many different values for each feature
sapply(test.data.raw, function(x) length(unique(x)))

# visualize missing data
library(Amelia)
missmap(test.data.raw, main = "Missing values vs observed")

#subset the data
testdata <- subset(test.data.raw, select=c(2,3,4,5))

# substitute missing data with average
testdata$F2[is.na(testdata$F2)] <- mean(testdata$F2, na.rm=T)

# check factors
is.factor(testdata$label)
is.factor(testdata$F1)
is.factor(testdata$F2)
is.factor(testdata$F3)

# check categorical variable encoding to understand the model
contrasts(testdata$label)
contrasts(testdata$F3)

# remove rows in F3 with NAs
testdata <- testdata[!is.na(testdata$F3),]
rownames(testdata) <- NULL

#training and testing data is already split via python
train = training.data.raw
test = test.data.raw

# Model fitting
model <- glm(label ~., family=binomial(link='logit'), data=train)
model
#measure predictability ### errors
#####################################
predict(model, newdata=subset(test,select=c(2,3,4,5)), type='response')
fitted.results <- ifelse(fitted.results > 0.5, 1, 0)
fitted.results
library(ROCR)
# ROC and AUC
p <- predict(model, newdata=subset(test,select=c(2,3,4,5)),type='response')
pr <- prediction(p, test$label)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)

auc <- performance(pr, measure = "auc")