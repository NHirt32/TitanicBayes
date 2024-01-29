from sklearn.linear_model import Perceptron
import dataProcessing as dp

perceptron = Perceptron()

def percFit(features, target):
    """
    This function fits our Perceptron model based off the features and targets of our test data

    Args:
        features (DataFrame): The features of our training data
        target (DataFrame): The targets of our training data
    Returns:
        Perceptron: A fitted Perceptron model
    """
    perceptron.fit(features, target.values.ravel())
    return perceptron

def percPredict(testFeatures, model):
    """
    This function utilizes a fitted Perceptron model and test features to predict test targets

    Args:
        testFeatures (DataFrame): The features of our test data that we are trying to predict our targets with
        model (Perceptron): Our Perceptron model that has already been fitted
    Returns:
        NumPy Array: The predicted targets of our test data
    """
    survPrediction = model.predict(testFeatures)
    return survPrediction


def prc(trainingData, testData):
    """
    This function encapsulates the entirety of the Perceptron model process including:
    Processing data -> Fitting our model -> Predicting our test data's targets -> Combining predictions and SurvivorIds

    Args:
        trainingData (DataFrame): The training data for our model
        testData (DataFrame): The test data for our model
    Returns:
        DataFrame: Our final DataFrame with survival prediction and PassengerId
    """
    features, target, noIdPrediction = dp.trainingSplit(trainingData, testData)
    fittedPerceptron = percFit(features, target)
    prediction = percPredict(noIdPrediction, fittedPerceptron)
    predictionCombined = dp.predictionCombine(prediction, testData)
    return predictionCombined