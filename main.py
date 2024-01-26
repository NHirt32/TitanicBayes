import pandas as pd
from sklearn.naive_bayes import GaussianNB
from dataProcessing import dataPreprocessing

# Removing the max column account to see all model features
pd.set_option('display.max_columns', None)


# Data processing
trainingData, testData = dataPreprocessing("train.csv", "test.csv")

# Checking for NaN
# print('NaN`s in DF:', testData.isna().any())

# Machine Learning
# Creating an X and Y to pass into NaiveBayes algorithm
features = trainingData.drop("Survived", axis=1)
target = trainingData[['Survived']]

naiveBayes = GaussianNB()
naiveBayes.fit(features, target.values.ravel())
survPrediction = naiveBayes.predict(testData)

prediction = pd.DataFrame({'PassengerId':testData.PassengerId, 'Survived':survPrediction})
print(prediction.head())
prediction.to_csv("prediction.csv", index=False)

