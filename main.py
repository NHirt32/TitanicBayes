import pandas as pd
import dataVisualization as dv
from dataProcessing import dataPreprocessing

from naiveBayes import nb
from perceptron import prc
from decisionTree import decTree

# Removing the max column account to see all model features
pd.set_option('display.max_columns', None)

# Data processing
trainingData, testData = dataPreprocessing("train.csv", "test.csv")

# Feature heatmap printing
dv.heatmap(trainingData)

# Dropping low impact features
lowImpactFeatures = ['SibSp', 'Parch']
for feature in lowImpactFeatures:
    trainingData.drop(feature, axis=1, inplace=True)
    testData.drop(feature, axis=1, inplace=True)


# Machine Learning
nbPrediction = nb(trainingData, testData)
nbPrediction.to_csv("nbPrediction.csv", index=False)

prcPrediction = prc(trainingData, testData)
prcPrediction.to_csv("prcPrediction.csv", index=False)

decTreePrediction = decTree(trainingData, testData)
decTreePrediction.to_csv("decTreePrediction.csv", index=False)