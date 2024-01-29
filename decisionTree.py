from sklearn.tree import DecisionTreeClassifier
import dataProcessing as dp

decisionTree = DecisionTreeClassifier()

def decTreeFit(features, target):
    decisionTree.fit(features, target.values.ravel())
    return decisionTree

def decTreePredict(testFeatures, model):
    survPrediction = model.predict(testFeatures)
    return survPrediction


def decTree(trainingData, testData):
    features, target, noIdPrediction = dp.trainingSplit(trainingData, testData)
    fittedDecTree = decTreeFit(features, target)
    prediction = decTreePredict(noIdPrediction, fittedDecTree)
    predictionCombined = dp.predictionCombine(prediction, testData)
    return predictionCombined