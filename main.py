import pandas as pd
from sklearn.naive_bayes import GaussianNB
import dataProcessing as dp

# Removing the max column account to see all model features
pd.set_option('display.max_columns', None)


# Data processing
trainingData, testData = dp.dataPreprocessing("train.csv", "test.csv")


# Machine Learning
# Creating an X and Y to pass into NaiveBayes algorithm
features = trainingData.drop("Survived", axis=1)
target = trainingData[['Survived']]
print(features.head())

naiveBayes = GaussianNB()
# naiveBayes.fit(features, target.values.ravel())
# survPrediction = naiveBayes.predict(testData)
#
# prediction = pd.DataFrame({'PassengerId':testData.PassengerId, 'Survived':survPrediction})
# print(prediction.head())
# Current Questions:
# What is SibSp, Parch, and Pclass
# How is Feature Engineering(Categorization) affected by min-max feature scaling?

