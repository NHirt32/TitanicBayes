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
testData.drop("Name", axis=1, inplace=True)
# Normalizing Sex to 0(male) or 1(female)
trainingData['Sex'] = trainingData['Sex'].map({'male': 0, 'female': 1})
testData['Sex'] = testData['Sex'].map({'male': 0, 'female': 1})
# Replacing all values in categorical columns with first character of each value if value is a non-empty string
trainingData['Cabin'] = trainingData['Cabin'].apply(lambda x: x[0] if isinstance(x,str) and len(x) > 0 else x)
testData['Cabin'] = testData['Cabin'].apply(lambda x: x[0] if isinstance(x,str) and len(x) > 0 else x)
trainingData['Ticket'] = trainingData['Ticket'].apply(lambda x: x[0] if isinstance(x,str) and len(x) > 0 else x)
testData['Ticket'] = testData['Ticket'].apply(lambda x: x[0] if isinstance(x,str) and len(x) > 0 else x)

# Filling missing data in columns
trainingData.ffill(inplace=True)
testData.ffill(inplace=True)

# Replacing categorical columns with new binary columns
trainingData = pd.get_dummies(trainingData, columns=['Ticket', 'Cabin', 'Embarked'])
testData = pd.get_dummies(testData, columns=['Ticket', 'Cabin', 'Embarked'])

# Applying min-max feature scaling to the following columns: PassengerId, Pclass, Age, SibSp, Parch, Fare
scalingFeatureList = ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
for feature in scalingFeatureList:
    trainingMax = trainingData[feature].max()
    testMax = testData[feature].max()
    trainingMin = trainingData[feature].min()
    testMin = testData[feature].min()
    trainingData[feature] = (trainingData[feature]-trainingMin)/(trainingMax-trainingMin)
    testData[feature] = (testData[feature] - testMin) / (testMax - testMin)


# Creating an X and Y to pass into NaiveBayes algorithm
features = trainingData.drop("Survived", axis=1)
target = trainingData[['Survived']]

# Converting boolean binary categories to numeric
features = features.astype('float32')

print(features.head())

naiveBayes.fit(features, target.values.ravel())
survPrediction = naiveBayes.predict(testData)

prediction = pd.DataFrame({'PassengerId':testData.PassengerId, 'Survived':survPrediction})
print(prediction.head())

# What is SibSp, Parch, and Pclass

