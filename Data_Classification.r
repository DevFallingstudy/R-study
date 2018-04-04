class_pack = c("caret",
               "skimr",
               "RANN",
               "randomForest",
               "fastAdaboost",
               "gbm",
               "xgboost",
               "caretEnsemble",
               "C50",
               "earth")
install.packages(class_pack)

library(caret) # total packages for data analysis
library(skimr) # Descriptive statistics
library(RANN) # Predicting missing values

orange = read.csv('https://raw.githubusercontent.com/selva86/datasets/master/orange_juice_withmissing.csv')

# Create the training and test dataset
trainRowNumbers = createDataPartition(orange$Purchase, p = 0.8, list = FALSE)
trainData = orange[trainRowNumbers, ]
testData = orange[-trainRowNumbers, ]

x = trainData[, 2:18]
y = trainData$Purchase

# Show descriptive statistics
skimmed = skim_to_wide(trainData)
skimmed[, c(10:16)]

# Impute missing values using preProcess - NA ?????????
anyNA(trainData)
preProcess_missingdata_model = preProcess(trainData, method='knnImpute')
trainData_impute = predict(preProcess_missingdata_model, newdata = trainData)
anyNA(trainData_impute)

# one-hot encoding for categorical variables - one-hot ????????? ??????
dummies_model = dummyVars(Purchase ~. , data = trainData_impute)
trainData_mat = predict(dummies_model, newdata = trainData_impute)
trainData_dummy = data.frame(trainData_mat)

# Preprocessing - mapping
preProcess_range_model = preProcess(trainData_dummy, method = 'range')
trainData_pre = predict(preProcess_range_model, newdata = trainData_dummy)
apply(trainData_pre[, 1:10], 2, FUN=function(x){c('min'=min(x), 'max'=max(x) )} )

trainData_pre$Purchase = y # Append the Y variables
# rm(trainData_dummy, trainData_impute, trainData_mat, trainRowNumbers) # Delete unuse data

# Fine what is important vatiables
model_rf = train(Purchase ~., data = trainData_pre, method = 'rf')
model_rf
varimp_rf = varImp(model_rf)
plot(varimp_rf)

# Neural Net
#model_nnet = train(Purchase ~., data = trainData_pre, method = 'nnet')
#model_nnet
#plot(model_nnet)

# Prepare the test dataset
testData2 = predict(preProcess_missingdata_model, testData)
testData3 = predict(dummies_model, testData2)
testData3 = data.frame(testData3)
testData4 = predict(preProcess_range_model, testData3)

# hyperparameter tuning
fitControl <- trainControl(
  method = 'cv',
  number = 5,
  savePredictions = 'final',
  classProbs = T,
  summaryFunction = twoClassSummary
)

model_mars2 = train(Purchase ~. , data = trainData_pre, method = 'earth', metric = 'ROC', tuneLength = 5, trControl=fitControl)
model_rf2 = train(Purchase ~. , data = trainData_pre, method = 'rf', metric = 'ROC', tuneLength = 5, trControl=fitControl)
model_svm2 = train(Purchase ~. , data = trainData_pre, method = 'svmRadial', metric = 'ROC', tuneLength = 5, trControl=fitControl)
model_nnet2 = train(Purchase ~. , data = trainData_pre, method = 'nnet', metric = 'ROC', tuneLength = 5, trControl=fitControl)

model_mars2
model_rf2
model_svm2
model_nnet2

predicted_mars2 = predict(model_mars2, testData4)
confusionMatrix(reference = testData$Purchase, data = predicted_mars2, mode = 'everything', positive = 'MM')

predicted_rf2 = predict(model_rf2, testData4)
confusionMatrix(reference = testData$Purchase, data = predicted_rf2, mode = 'everything', positive = 'MM')

predicted_svm2 = predict(model_svm2, testData4)
confusionMatrix(reference = testData$Purchase, data = predicted_svm2, mode = 'everything', positive = 'MM')

predicted_nnet2 = predict(model_nnet2, testData4)
confusionMatrix(reference = testData$Purchase, data = predicted_nnet2, mode = 'everything', positive = 'MM')
