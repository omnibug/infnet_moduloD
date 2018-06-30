# Install packages
#install.packages('randomForest')
#install.packages('ggthemes')
#install.packages('caret')
#install.packages('curl-config')
#install.packages('RCurl')
#install.packages('Metrics')
#install.packages('xgboost')
#install.packages('mice')
#install.packages('dplyr')
# Load packages
library('ggplot2') # visualization
library('lattice') 
library(caret) # for dummyVars
#library(RCurl) # download https data
library(Metrics) # calculate errors
library(xgboost) # model
library(caret) # for dummyVars
library(Rcpp)
library(mice)
#library('ggthemes') # visualization
#library('scales') 
library('dplyr') # data manipulation

###############################################################################

trainfile <- read.csv(file=file.path(getwd(),'MEGAsync','MIT','myosotis','myosotis_database_encoded.csv'), stringsAsFactors=FALSE)
print(names(trainfile))
#trains <- c('SeriousDlqin2yrs','NumberOfTime30.59DaysPastDueNotWorse','NumberOfTime60.89DaysPastDueNotWorse','NumberOfTimes90DaysLate','age','NumberOfOpenCreditLinesAndLoans','NumberOfDependents','NumberRealEstateLoansOrLines')
trains <- c('istatus',
            'icor_da_pele', 
            'isexo', 
            'iolhos', 
            'idade', 
            'dias_desaparecido'
            )

trainPortion <- floor(nrow(trainfile)*0.7)
print(trainPortion)
# Split the data back into a train set and a test set
trainf <- trainfile[1:trainPortion,]
testf <- trainfile[trainPortion:nrow(trainfile),]
md.pattern(train)

train <- trainf[trains]
test <- testf[trains]
#md.pattern(train)

# binarize all factors
library(caret)
dmy <- dummyVars(" ~ .", data = train)
trainTrsf <- data.frame(predict(dmy, newdata = train))
dmy <- dummyVars(" ~ .", data = test)
testTrsf <- data.frame(predict(dmy, newdata = test))
###############################################################################

# what we're trying to predict istatus
outcomeName <- c('istatus')
# list of features
predictors <- names(trainTrsf)[!names(trainTrsf) %in% outcomeName]
predictorsub <- names(testTrsf)[!names(testTrsf) %in% outcomeName]

# play around with settings of xgboost - eXtreme Gradient Boosting (Tree) library
# https://github.com/tqchen/xgboost/wiki/Parameters
# max.depth - maximum depth of the tree
# nrounds - the max number of iterations

trainSet <- trainTrsf
testSet <- testTrsf
bst <- xgboost(data = as.matrix(trainSet[,predictors]), label = trainSet[,outcomeName], max.depth=2, nround=60000, objective = "reg:linear", verbose=1, eta=0.001, min_child_weight=1, gamma=0, subsample=0.1, colsample_bytree=0.5, missing=NaN)
pred <- predict(bst, as.matrix(testSet[,predictors]), outputmargin=TRUE, missing=NaN)

predsub <- round(pred,0)
submission <- data.frame(id = testf$id, istatus = testf$istatus)
submission$Prediction <- predsub

#write.csv(submission, file=file.path(getwd(),'MEGAsync','MIT','myosotis','myosotis_database_prediction_linux_4.csv'), row.names=FALSE)

