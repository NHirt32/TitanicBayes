import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB

# Removing the max column account to see all model features
pd.set_option('display.max_columns', None)

trainingData = pd.read_csv('train.csv')
testData = pd.read_csv('test.csv')

naiveBayes = GaussianNB()

# Dropping irrelevant column 'Name'
trainingData.drop("Name", axis=1, inplace=True)
# Normalizing Sex to 0(male) or 1(female)
trainingData['Sex'] = trainingData['Sex'].map({'male': 0, 'female': 1})

# Temp drop on categorical data
trainingData.drop(['Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
trainingData.ffill(inplace=True)
# Temp processing of test data
testData.drop("Name", axis=1, inplace=True)
testData['Sex'] = testData['Sex'].map({'male': 0, 'female': 1})
testData.drop(['Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
testData.ffill(inplace=True)

# Replacing categorical columns with new binary columns
# trainingData = pd.get_dummies(trainingData, columns=['Ticket', 'Cabin', 'Embarked'])


# Creating an X and Y to pass into NaiveBayes algorithm
features = trainingData.drop("Survived", axis=1)
target = trainingData[['Survived']]

# features = features.astype('float32')

print(features.head())

naiveBayes.fit(features, target.values.ravel())
survPrediction = naiveBayes.predict(testData)

prediction = pd.DataFrame({'PassengerId':testData.PassengerId, 'Survived':survPrediction})
print(prediction.head())

# What is SibSp, Parch, and Pclass

