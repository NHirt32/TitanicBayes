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

# Creating an X and Y to pass into NaiveBayes algorithm
features = trainingData[['PassengerId', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']].copy()
target = trainingData['Survived'].copy()

naiveBayes.fit(features, target)
naiveBayes.predict(testData)

# Potential categorization ideas include collapsing age, cabin section, fares
# Maybe drop Embarked?
# What is SibSp, Parch, and Pclass
prediction = pd.DataFrame({"PassengerId":features.PassengerId, "Survived":target})
print(prediction.head())
