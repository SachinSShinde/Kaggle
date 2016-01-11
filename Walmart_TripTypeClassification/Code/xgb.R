### Walmart Trip Type Classification 
### https://www.kaggle.com/c/walmart-recruiting-trip-type-classification
### Author: Sachin Shinde
### Date: 12/27/2015

## setting working directory
path <- '' #Set path as required
setwd(path)
getwd()

## loading libraries
require(data.table)
require(xgboost)
require(dummies)
require(dplyr)
require(plyr)
set.seed(2712)

## loading data (edit the paths)
trainTrip <- read.csv("./train.csv", stringsAsFactors=T)
testTrip <- read.csv("./test.csv", stringsAsFactors=T)

# Function for number of returned items
Returned <- function(x){
  if(x <= 0){
    return (abs(x))
  }else
    return (0)
}

# Feature representing Returned items 
trainTrip$Returned <- sapply(trainTrip[,5], Returned)
testTrip$Returned <- sapply(testTrip[,4], Returned)

# Assigning constant(9999) to missing values in FinelineNumber
trainTrip[is.na(trainTrip$FinelineNumber),7] <- 9999
testTrip[is.na(testTrip$FinelineNumber), 6] <- 9999


# onehot-encoding DepartmentDescription, FinelineNumber and Weekday
X_trainTrip <- dummy.data.frame(trainTrip, names=c("DepartmentDescription"), sep="_")
X_testTrip <- dummy.data.frame(testTrip, names=c("DepartmentDescription"), sep="_")

X_trainTrip <- dummy.data.frame(X_trainTrip, names=c("Weekday"), sep="_")
X_testTrip <- dummy.data.frame(X_testTrip, names=c("Weekday"), sep="_")

X_trainTrip <- dummy.data.frame(X_trainTrip, names=c("FinelineNumber"), sep="_")
X_testTrip <- dummy.data.frame(X_testTrip, names=c("FinelineNumber"), sep="_")

# Aggregating all values as per TripType and VisitNumber for Training Dataset
x_tr <- X_trainTrip %>% group_by(TripType, VisitNumber) %>% summarise_each(funs(sum))
x_te <- X_testTrip %>% group_by(VisitNumber) %>% summarise_each(funs(sum))

# lables 
tlabels <- as.numeric(as.factor(x_tr$TripType))-1 #xgboost take features in [0,numOfClass)
x_tr <- subset(x_tr, select = -c(TripType))

# Keeping only those columns whiich are in Test dataset as it has less Department Description than Training dataset
col_name <- colnames(x_te)
x_tr <- subset(x_tr, select = c(col_name))

#ncol(x_te)
#ncol(x_tr)

# preparing data in matrix format
xgbtrain <- xgb.DMatrix(as.matrix(sapply(x_tr, as.numeric)), label=tlabels, missing=NA)

param <- list(objective = "multi:softprob", eta = 0.03, max.depth = 10, gamma = 0.5, min_child_weight = 4, 
     subsample = 1.0, colsample_bytree = 0.3, num_parallel_tree = 4, eval_metric = "mlogloss", 
     num_round = 500, num_class = 38)
nround = 500

# train xgboost
bst <- xgboost(data = xgbtrain, label = tlabels, missing = NA, 
               params = param,
               nrounds=nround, nthread = 8, booster = "gbtree", verbose = 1)

xgbtest <- xgb.DMatrix(data.matrix(sapply(x_te, as.numeric)), missing=NA)
pred_xgb = predict(bst,xgbtest)

probs_xgb <- t(matrix(pred_xgb, nrow=38, ncol=length(pred_xgb)/38))
typenames <- fread("./tts.csv") # Using already created tts.csv file with headers for submission
colnames(probs_xgb) <- typenames[,tt]
submit <- cbind(x_te[,1], probs_xgb)
col_name <- c("VisitNumber", colnames(probs_xgb[,]))
colnames(submit) <- col_name

## submission file

write.csv(submit,file='/Users/sachinshinde/SachinSShinde/My_Data/Study/Kaggle/Kaggle/Walmart_TripTypeClassification/Submission/submit.csv', quote=FALSE,row.names=FALSE)
zip(zipfile = '/Users/sachinshinde/SachinSShinde/My_Data/Study/Kaggle/Kaggle/Walmart_TripTypeClassification/Submission/submit', files = '/Users/sachinshinde/SachinSShinde/My_Data/Study/Kaggle/Kaggle/Walmart_TripTypeClassification/Submission/submit.csv')


## model performance (mlogloss)

# Public LB = 0.84584
# Private LB = 0.83062
