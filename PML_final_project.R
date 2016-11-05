
##############################################
# Practical Machine Learning: Final Project
# Developer: Meng Qiu
# Date: 10/30/2016
##############################################

#### Data Processing ####

## 1. Load libraries
library(caret); library(rattle); library(rpart); library(rpart.plot)
library(randomForest); library(repmis)

## 2. Import datasets
getwd()
setwd("C:/Users/Owen/Desktop/final project")
training <- read.csv("C:/Users/Owen/Desktop/final project/datasets/pml-training.csv", na.strings = c("NA", ""))
testing <- read.csv("C:/Users/Owen/Desktop/final project/datasets/pml-testing.csv", na.strings = c("NA", ""))
dim(training); dim(testing)
names(training); names(testing)
# The training dataset has 19622 observations and 160 variables, and the testing data set contains 20 observations 
# and the same variables as the training set. 
# We are to predict the outcome of the variable "classe" in the training set.

## 3. Data cleaning
# delete columns that contain missing values
training <- training[, -c(which(colSums(is.na(training))>1))]
testing <- testing[, -c(which(colSums(is.na(testing))>1))]
  # or, training <- training[, colSums(is.na(training)) == 0]
  # or, testing <- testing[, colSums(is.na(testing)) == 0]

# delete the first seven predictors which have little predicting power
trainData <- training[, -c(1:7)]
testData <- testing[, -c(1:7)]
# The cleaned data sets, trainData and testData, both have 53 columns with the same first 52 variables 
# and the last variable classe and "problem_id" individually. 
# The trainData has 19622 rows while testData has 20 rows.

## 4. Data spliting
# In order to get out-of-sample errors, we split the cleaned training set trainData into a training set (train, 70%) 
# for prediction and a validation set (valid 30%) to compute the out-of-sample errors.
set.seed(1234) 
inTrain <- createDataPartition(trainData$classe, p = 0.7, list = FALSE)
train <- trainData[inTrain, ]
valid <- trainData[-inTrain, ]


#### Prediction Algorithm - Classification Tree & Random Forest ####

## 1. Classification trees
control <- trainControl(method = "cv", number = 5)
fit_rpart <- train(classe ~ ., data = train, method = "rpart", trControl = control)
print(fit_rpart, digits = 4)

fancyRpartPlot(fit_rpart$finalModel, main="classification tree plot", tweak=1)


# predict outcomes using validation set
predict_rpart <- predict(fit_rpart, valid)
# show prediction result
(conf_rpart <- confusionMatrix(valid$classe, predict_rpart))
# show prediction accuracy
(accuracy_rpart <- conf_rpart$overall[1])

# From the confusion matrix, the accuracy rate is 0.489, and so the out-of-sample error rate is 0.511. 
# Using classification tree did not predict the outcome classe very well.

## 2. Random forest
# Since the classification tree didn't perform well, we'll turn to random forest method.
set.seed(12345)
fit_rf <- train(classe ~ ., data = train, method = "rf", trControl = control)
print(fit_rf, digits = 4)

predictRf <- predict(fit_rf, valid)
cm_rf <- confusionMatrix(valid$classe, predictRf)
(accuracy_rf <- cm_rf$overall[1])  # 0.9942226 
plot(cm_rf$table, col = cm_rf$byClass, main = paste("Random Forest Confusion Matrix: Accuracy =", round(cm_rf$overall['Accuracy'], 4)))

# For this dataset, random forest method is way better than classification tree method. The accuracy rate is 0.994, and so the out-of-sample error rate is 0.006. 
# This may be due to the fact that many predictors are highly correlated. Random forests chooses a subset of predictors at each split and decorrelate the trees. 
# This leads to high accuracy, although this algorithm is sometimes difficult to interpret and computationally inefficient.

#### Prediction on the Test Dataset ####
predict(fit_rf, testData)
# project ends


library(knitr)
library(rmarkdown)

