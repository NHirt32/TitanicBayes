from sklearn.tree import DecisionTreeClassifier
import dataProcessing as dp

decisionTree = DecisionTreeClassifier()

def decTreeFit(features, target):
    """
    This function fits our Decision Tree model based off the features and targets of our test data

    Args:
        features (DataFrame): The features of our training data
        target (DataFrame): The targets of our training data
    Returns:
        DecisionTreeClassifier: A fitted Decision Tree model
    """
    decisionTree.fit(features, target.values.ravel())
    return decisionTree

def decTreePredict(testFeatures, model):
    """
    This function utilizes a fitted Decision Tree model and test features to predict test targets

    Args:
        testFeatures (DataFrame): The features of our test data that we are trying to predict our targets with
        model (DecisionTreeClassifier): Our Decision Tree model that has already been fitted
    Returns:
        NumPy Array: The predicted targets of our test data
    """
    survPrediction = model.predict(testFeatures)
    return survPrediction


def decTree(trainingData, testData):
    """
    This function encapsulates the entirety of the Decision Tree model process including:
    Processing data -> Fitting our model -> Predicting our test data's targets -> Combining predictions and SurvivorIds

    Args:
        trainingData (DataFrame): The training data for our model
        testData (DataFrame): The test data for our model
    Returns:
        DataFrame: Our final DataFrame with survival prediction and PassengerId
    """
    features, target, noIdPrediction = dp.trainingSplit(trainingData, testData)
    fittedDecTree = decTreeFit(features, target)
    prediction = decTreePredict(noIdPrediction, fittedDecTree)
    predictionCombined = dp.predictionCombine(prediction, testData)
    return predictionCombined