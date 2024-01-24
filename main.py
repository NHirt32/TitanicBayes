import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder # For replacing non-numeric data with numeric data
from sklearn.impute import SimpleImputer # For filling missing values with mean
from sklearn.compose import ColumnTransformer # For applying encoder to dataframe columns

# Removing the max column account to see all model features
pd.set_option('display.max_columns', None)

trainingData = pd.read_csv('train.csv')
testData = pd.read_csv('test.csv')

naiveBayes = GaussianNB()

# Checking dataframe properties to scan for NaN values
print(trainingData.info())
# NaN values in Age, Cabin, and Embarked.
# Creating encoder to replace non-numeric values with numeric values
le = LabelEncoder()
nonNumeric = ['Cabin', 'Ticket', '']
trainingData['Cabin'] = le.fit_transform(trainingData['Cabin'])

# Dropping irrelevant column 'Name', in the future we will apply regex's to extract Titles
trainingData.drop("Name", axis=1, inplace=True)
testData.drop("Name", axis=1, inplace=True)
# Normalizing Sex to 0(male) or 1(female)
trainingData['Sex'] = trainingData['Sex'].map({'male': 0, 'female': 1})
testData['Sex'] = testData['Sex'].map({'male': 0, 'female': 1})
# Replacing all values in categorical columns with first character of each value if value is a non-empty string
# trainingData['Cabin'] = trainingData['Cabin'].apply(lambda x: x[0] if isinstance(x,str) and len(x) > 0 else x)
# testData['Cabin'] = testData['Cabin'].apply(lambda x: x[0] if isinstance(x,str) and len(x) > 0 else x)
# trainingData['Ticket'] = trainingData['Ticket'].apply(lambda x: x[0] if isinstance(x,str) and len(x) > 0 else x)
# testData['Ticket'] = testData['Ticket'].apply(lambda x: x[0] if isinstance(x,str) and len(x) > 0 else x)


# Filling missing data in columns
trainingData.ffill(inplace=True)
testData.ffill(inplace=True)

# Replacing categorical columns with new binary columns
# trainingData = pd.get_dummies(trainingData, columns=['Ticket', 'Cabin', 'Embarked'])
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

# Converting boolean binary categories to numeric (For get_dummies, may drop)
# features = features.astype('float32')

print(features.head())

# Machine Learning

# naiveBayes.fit(features, target.values.ravel())
# survPrediction = naiveBayes.predict(testData)
#
# prediction = pd.DataFrame({'PassengerId':testData.PassengerId, 'Survived':survPrediction})
# print(prediction.head())

# What is SibSp, Parch, and Pclass

