
# Load the raw training data and replace missing values with NA
training.data.raw <- read.csv('SpamProcessedData.csv', header=T, na.strings=c(""))
str(training.data.raw)

# Output the number of missing values for each column
sapply(training.data.raw,function(x) sum(is.na(x)))

# Quick check for how many different values for each feature
sapply(training.data.raw, function(x) length(unique(x)))

# A visual way to check for missing data
library(Amelia)
missmap(training.data.raw, main = "Missing values vs observed")

# Subsetting the data
data <- subset(training.data.raw,select=c(2,3,4,5,6))

# Substitute the missing values with the average value
data$Website[is.na(data$Website)] <- mean(data$Website,na.rm=T)
data$W.count[is.na(data$W.count)] <- mean(data$W.count, na.rm=T)
data$F.WordCount[is.na(data$F.WordCount)] <- mean(data$F.WordCount, na.rm=T)
data$Spamword[is.na(data$Spamword)] <- mean(data$Spamword, na.rm=T)


# categorical variables
is.factor(data$label)         


# Check categorical variables encoding for better understanding of the fitted model
contrasts(data$label)        
    

# Remove rows with NAs
data <- data[!is.na(data$Website),]
data <- data[!is.na(data$W.count),]
data <- data[!is.na(data$F.WordCount),]
data <- data[!is.na(data$Spamword),]
rownames(data) <- NULL

# Train test splitting
train <- data[1:2786,]
test <- data[2787:5574,]

# Model fitting
model <- glm(label ~.,family=binomial(link='logit'),data=train)

#-------------------------------------------------------------------------------
# MEASURING THE PREDICTIVE ABILITY OF THE MODEL
# If prob > 0.5 then 1, else 0. Threshold can be set for better results
fitted.results <- predict(model,newdata=subset(test,select=c(2,3,4,5)),type='response')
fitted.results <- ifelse(fitted.results > 0.5,"spam","ham")


library(ROCR)
# ROC and AUC
p <- predict(model, newdata=subset(test,select=c(2,3,4,5)), type="response")
pr <- prediction(p, test$label)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)

auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

# Confusion matrix
library(caret)
library(e1071)
# TPR = sensitivity (e.g., How many relevant items are selected by the model?)
# FPR = 1-specificity (e.g., how many non-relevant items are truly negatively selected by the mode?)
reference=factor(test$label)
str(reference)
str(fitted.results)

confusionMatrix(data=factor(fitted.results),reference, positive=levels(reference)[2])

# Student Exercise
# 1) Calculate the TPR and FPR using the confusion matrix
#    What is TPR? .955
#    What is FPR? .295
#
# 3) Homework! Do a same job above with the binomial logistic regression 
#    w/ the 10 folds Cross Validation 
#    (Hint: You might check a caret package.) 
