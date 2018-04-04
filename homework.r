# e-mail -> sunnybenlim@gmail.com
# ???????????? : train&test & preprocess & tuning & confusion matrix
# ???????????? : Confusion matrix ??????(??????)
# output : dataset$class : {good, bad}
# evaluation

class_pack = c("caret",
               "skimr",
               "RANN",
               "randomForest",
               "fastAdaboost",
               "gbm",
               "xgboost",
               "caretEnsemble",
               "C50",
               "earth",
               "ada")
install.packages(class_pack)

library(caret) # total packages for data analysis
library(skimr) # Descriptive statistics
library(RANN) # Predicting missing values
library(caretEnsemble)

url_address = "https://www.dropbox.com/s/4wpkhme7476zdt3/dataset.csv?dl=1"
dataset <- read.csv(url_address)

# Create the training and test dataset
trainRowNumbers = createDataPartition(dataset$class, p = 0.8, list = FALSE)
trainData = dataset[trainRowNumbers, ]
testData = dataset[-trainRowNumbers, ]

x = trainData[, 1:20]
y = trainData$class

# Impute missing values using preProcess - NA ?????????
anyNA(trainData)
preProcess_missingdata_model = preProcess(trainData, method='knnImpute')
trainData_impute = predict(preProcess_missingdata_model, newdata = trainData)
anyNA(trainData_impute)

# one-hot encoding
dummies_model = dummyVars(class ~. , data = trainData_impute)
trainData_mat = predict(dummies_model, newdata = trainData_impute)
trainData_dummy = data.frame(trainData_mat)

# Preprocessing - mapping
preProcess_range_model = preProcess(trainData_dummy, method = 'range')
trainData_pre = predict(preProcess_range_model, newdata = trainData_dummy)
apply(trainData_pre[, 1:10], 2, FUN=function(x){c('min'=min(x), 'max'=max(x) )} )

trainData_pre$class = y # Append the Y variables

#trainData_dummy$class = y # Append the Y variables

# Fine what is important vatiables
#model_rf = train(class ~., data = trainData_pre, method = 'rf')
#model_rf
#varimp_rf = varImp(model_rf)
#plot(varimp_rf)

# Prepare the test dataset
testData2 = predict(preProcess_missingdata_model, testData)
testData3 = predict(dummies_model, testData2)
testData3 = data.frame(testData3)
testData4 = predict(preProcess_range_model, testData3)


# Ensemble

trainControl <- trainControl(method="repeatedcv", 
                             number=10, 
                             repeats=3,
                             savePredictions=TRUE, 
                             classProbs=TRUE)

algorithmList <- c('rf', 'nnet', 'earth', 'xgbDART', 'svmRadial')
models <- caretList(class ~ ., data=trainData_pre, trControl=trainControl, methodList=algorithmList) 
results <- resamples(models)
summary(results)
bwplot(results, scales = list(x=list(relation = "free"), y=list(relation="free")))

model_ensemble = caretEnsemble(models)
predicted_ensemble = predict(model_ensemble, testData4)
confusionMatrix(reference = testData$class, data = predicted_ensemble, mode = 'everything')


# hyperparameter tuning
fitControl <- trainControl(
  method = 'repeatedcv',
  number = 10,
  savePredictions = 'final',
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

model_mars2 = train(class ~. , data = trainData_pre, method = 'earth', metric = 'ROC', tuneLength = 5, trControl=fitControl)
model_rf2 = train(class ~. , data = trainData_pre, method = 'rf', metric = 'ROC', tuneLength = 5, trControl=fitControl)
model_svm2 = train(class ~. , data = trainData_pre, method = 'svmRadial', metric = 'ROC', tuneLength = 5, trControl=fitControl)
model_nnet2 = train(class ~. , data = trainData_pre, method = 'nnet', metric = 'ROC', tuneLength = 5, trControl=fitControl)
model_ada = train(class ~. , data = trainData_pre, method = 'ada', metric = 'ROC', tuneLength = 5, trControl=fitControl)

model_mars2
model_rf2
model_svm2
model_nnet2

predicted_mars2 = predict(model_mars2, testData4)
confusionMatrix(reference = testData$class, data = predicted_mars2, mode = 'everything')

predicted_rf2 = predict(model_rf2, testData4)
confusionMatrix(reference = testData$class, data = predicted_rf2, mode = 'everything')

predicted_svm2 = predict(model_svm2, testData4)
confusionMatrix(reference = testData$class, data = predicted_svm2, mode = 'everything')

predicted_nnet2 = predict(model_nnet2, testData4)
confusionMatrix(reference = testData$class, data = predicted_nnet2, mode = 'everything')

predicted_ada = predict(model_ada, testData4)
confusionMatrix(reference = testData$class, data = predicted_ada, mode = 'everything')
