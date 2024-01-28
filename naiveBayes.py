from sklearn.naive_bayes import GaussianNB
import pandas as pd

naiveBayes = GaussianNB()

def nbFit(features, target):
    """
    This function fits our Naive Bayes model based off the features and targets of our test data

    Args:
        features (DataFrame): The features of our training data
        target (DataFrame): The targets of our training data
    Returns:
        GaussianBayes: A fitted Naive Bayes model
    """
    naiveBayes.fit(features, target.values.ravel())
    return naiveBayes

def nbPredict(testFeatures, model):
    """
    This function utilizes a fitted Naive Bayes model and test features to predict test targets

    Args:
        testFeatures (DataFrame): The features of our test data that we are trying to predict our targets with
        model (Naive Bayes): Our Naive Bayes model that has already been fitted
    Returns:
        NumPy Array: The predicted targets of our test data
    """
    survPrediction = model.predict(testFeatures)
    return survPrediction

def nbProcess(trainingData, testData):
    """
    This function seperates our training and testing datasets to be appropriately used by our model

    Args:
        trainingData (DataFrame): Our initial unprocessed set of training data
        testData (DataFrame): Our initial unprocessed set of testing data
    Returns:
        tuple (DataFrame, DataFrame, DataFrame): Processed Dataframes ready to be used by our model
    """
    features = trainingData.drop({"Survived", "PassengerId"}, axis=1)
    target = trainingData[['Survived']]
    noIdPrediction = testData.drop('PassengerId', axis=1)
    return features, target, noIdPrediction

def predictionCombine(prediction, testData):
    """
    This function combines the unused test SurvivorId's and our predicted Survivals into a singular DataFrame

    Args:
        prediction (DataFrame): Our predicted values for survival
        testData (DataFrame): The DataFrame of our testing data
    Returns:
        DataFrame: Final DataFrame with paired survival prediction and PassengerId
    """
    prediction = pd.DataFrame({'PassengerId': testData.PassengerId, 'Survived': prediction})
    return prediction

def nb(trainingData, testData):
    """
    This function encapsulates the entirety of the Naive Bayes model process including:
    Processing data -> Fitting our model -> Predicting our test data's targets -> Combining predictions and SurvivorIds

    Args:
        trainingData (DataFrame): The training data for our model
        testData (DataFrame): The test data for our model
    Returns:
        DataFrame: Our final DataFrame with survival prediction and PassengerId
    """
    features, target, noIdPrediction = nbProcess(trainingData, testData)
    fittedBayes = nbFit(features, target)
    prediction = nbPredict(noIdPrediction, fittedBayes)
    predictionCombined = predictionCombine(prediction, testData)
    return predictionCombined
