from sklearn.linear_model import Perceptron
import dataProcessing as dp

perceptron = Perceptron()

def percFit(features, target):
    perceptron.fit(features, target.values.ravel())
    return perceptron

def percPredict(testFeatures, model):
    survPrediction = model.predict(testFeatures)
    return survPrediction


def prc(trainingData, testData):
    features, target, noIdPrediction = dp.trainingSplit(trainingData, testData)
    fittedPerceptron = percFit(features, target)
    prediction = percPredict(noIdPrediction, fittedPerceptron)
    predictionCombined = dp.predictionCombine(prediction, testData)
    return predictionCombined