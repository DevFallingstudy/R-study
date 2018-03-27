library(caret)
library(earth)

data_raw = iris

trainRowNumbers = createDataPartition(data_raw$Species, p = 0.8, list = FALSE)
trainData = data_raw[trainRowNumbers, ]
testData = data_raw[-trainRowNumbers, ]

model = train(Species ~. , data = trainData, method = 'rf')
predictedData = predict(model, testData)

sum(predictedData == testData$Species) / 30 * 100