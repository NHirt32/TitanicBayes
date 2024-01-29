import pandas as pd
import dataVisualization as dv
from dataProcessing import dataPreprocessing
from dataProcessing import testMin, testMax

from naiveBayes import nb
from perceptron import prc
from decisionTree import decTree

# Removing the max column account to see all model features
pd.set_option('display.max_columns', None)

# Data processing
trainingData, testData = dataPreprocessing("train.csv", "test.csv")

# dv.heatmap(trainingData)
# lowImpactFeatures = ['SibSp', 'Parch']
# for feature in lowImpactFeatures:
#     trainingData.drop(feature, axis=1, inplace=True)
#     testData.drop(feature, axis=1, inplace=True)

# Checking for NaN
# print('NaN`s in DF:', testData.isna().any())


# Machine Learning

# nbPrediction = nb(trainingData, testData)
# nbPrediction.to_csv("predictionNB.csv", index=False)

# prcPrediction = prc(trainingData, testData)
# prcPrediction.to_csv("prcPrediction.csv", index=False)

decTreePrediction = decTree(trainingData, testData)
decTreePrediction.to_csv("decTreePrediction.csv", index=False)