import pandas as pd
import dataVisualization as dv
from naiveBayes import nb
from dataProcessing import dataPreprocessing
from dataProcessing import testMin, testMax

# Removing the max column account to see all model features
pd.set_option('display.max_columns', None)

# Data processing
trainingData, testData = dataPreprocessing("train.csv", "test.csv")

# dv.heatmap(trainingData)
lowImpactFeatures = ['SibSp', 'Parch']
for feature in lowImpactFeatures:
    trainingData.drop(feature, axis=1, inplace=True)
    testData.drop(feature, axis=1, inplace=True)

# Checking for NaN
# print('NaN`s in DF:', testData.isna().any())

# Machine Learning

nbPrediction = nb(trainingData, testData)
nbPrediction.to_csv("prediction.csv", index=False)

