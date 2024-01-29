import pandas as pd
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', None)

nonNumericColumns = ['Cabin', 'Ticket', 'Embarked', 'Sex']
missingValueColumns = ['Age', 'Cabin', 'Embarked', 'Name', 'Fare']
scalableFeatureList = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked', 'Ticket', 'Name', 'PassengerId']
testMin = 0
testMax = 0

def fileRead(trainFile, testFile):
    """
    This function converts two .csv files into pandas dataframes

    Args:
        trainFile (.csv): Relative file path of training .csv file
        testFile (.csv): Relative file path of testing .csv file
    Returns:
        tuple: Returns a tuple containing both .csv files transformed into pandas dataframes
    """
    trainingData = pd.read_csv(trainFile)
    testData = pd.read_csv(testFile)
    return trainingData, testData

def encoding(trainingData, testData):
    """
    This function encodes non-numeric columns of two dataframes into numeric columns

    Args:
        trainingData (DataFrame): Training data of machine learning model in Pandas DataFrame format
        testData (DataFrame): Test Data for machine learning model in Pandas DataFrame format
    Returns:
        tuple: Returns a tuple of training and testing dataframes with non-numeric data encoded into numeric data
    """
    le = LabelEncoder()
    for column in nonNumericColumns:
        trainingData[column] = le.fit_transform(trainingData[column])
        testData[column] = le.fit_transform(testData[column])
    return trainingData, testData

def impute(trainingData, testData):
    """
    This function imputes missing column values of our machine learning dataframes with medians of said columns

    Args:
        trainingData (DataFrame): Training data of machine learning model in Pandas DataFrame format
        testData (DataFrame): Test Data for machine learning model in Pandas DataFrame format
    Returns:
        tuple: Returns a tuple of training and testing dataframes where
               all missing values in columns have been imputed with the existing median of the column
    """
    for column in missingValueColumns:
        columnMean = trainingData[column].mean()
        trainingData.fillna({column: columnMean}, inplace=True)
        columnMean = testData[column].mean()
        testData.fillna({column: columnMean}, inplace=True)
    return trainingData, testData

def scaling(trainingData, testData):
    """
    This function utilizes min-max scaling to scale all numeric values in our two dataframes

    Args:
        trainingData (DataFrame): Training data of machine learning model in Pandas DataFrame format
        testData (DataFrame): Test Data for machine learning model in Pandas DataFrame format
    Returns:
        tuple: Returns a tuple of training and testing dataframes where all numeric columns have been scaled between 0 - 1
    """
    for feature in scalableFeatureList:
        trainingMax = trainingData[feature].max()
        testMax = testData[feature].max()
        trainingMin = trainingData[feature].min()
        testMin = testData[feature].min()
        trainingData[feature] = (trainingData[feature] - trainingMin) / (trainingMax - trainingMin)
        if feature != 'PassengerId':
            testData[feature] = (testData[feature] - testMin) / (testMax - testMin)
    return trainingData, testData

def nameConversion(trainingData, testData):
    """
    This function will identify titles within 'Name' column entries in our two dataframes and convert
    categorical name data to numerical based on their title

    Args:
        trainingData (DataFrame): Training data of machine learning model in Pandas DataFrame format
        testData (DataFrame): Test Data for machine learning model in Pandas DataFrame format
    Returns:
        tuple: Returns a tuple of training and testing dataframes where the name columns has been converted to numeric
               based on people's titles
    """
    titleMapping = {
        'Mr.': 0,
        'Mrs.': 1,
        'Master': 2,
        'Miss.': 3,
        'Don': 4,
        'Rev.': 5

    }

    trainingData['Name'] = trainingData['Name'].apply(lambda x: next((titleMapping[title] for title in titleMapping if title in x), None))
    testData['Name'] = testData['Name'].apply(lambda x: next((titleMapping[title] for title in titleMapping if title in x), None))


    return trainingData, testData

def dataPreprocessing(trainingFile, testingFile):
    """
    This function utilizes various other dataprocessing functions the ordering of this processing is: Reading the
    data into pandas DataFrames -> Encoding non-numeric values -> Encode names based on title ->
    -> imputing missing values -> min-max scaling numeric

    values

    Args:
        trainingFile (.csv): .csv file containing our ML-model's training data
        testingFile (.csv): .csv file containing our ML-model's training data
    Returns:
        tuple: Returns two DataFrames which have been encoded, imputed, and scaled
    """
    trainingData, testData = fileRead(trainingFile, testingFile)
    trainingData, testData = encoding(trainingData, testData)
    trainingData, testData = nameConversion(trainingData, testData)
    trainingData, testData = impute(trainingData, testData)
    trainingData, testData = scaling(trainingData, testData)
    return trainingData, testData

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

def trainingSplit(trainingData, testData):
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
