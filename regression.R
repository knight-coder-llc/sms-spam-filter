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


