import math
import pandas as pd
import decorators as decor

def splitData(dataFrame, trainingRatio):
    ''' '''
    trainingSize = math.floor(len(dataFrame) * trainingRatio)
    testSize = len(dataFrame) - trainingSize
    training = dataFrame.sample(n=trainingSize, replace=False)
    test = dataFrame.drop(training.index)
    return training, test

def computeError(predictions, actuals):
    ''' '''
    assert len(predictions) == len(actuals), "Number of predictions and actuals must match"
    assert type(predictions) == type(actuals), "Type of predictions and actuals must match"
    misClassified = 0
    for i in predictions.index:
        if (predictions[i] != actuals[i]):
            misClassified += 1
    return 1 - ((len(actuals) - misClassified) / len(actuals))

def buildConfusionMatrix(predictions, actuals, features):
    ''' '''
    assert len(predictions) == len(actuals), "Number of predictions and actuals must match"
    assert type(predictions) == type(actuals), "Type of predictions and actuals must match"
    table = {}
    features.sort()
    
    # Initialize the table with column header and cell values with 0
    for feature in features:
        table[feature] = [0 for i in range(len(features))]
    
    matrix = pd.DataFrame(data=table, index=features)
    
    # Count the misclassifications
    for i in predictions.index:
        if (predictions[i] == actuals[i]):
            matrix.loc[predictions[i], predictions[i]] += 1
        else:
            matrix.loc[actuals[i], predictions[i]] += 1

    # Rename column names and row indeces for clarity
    renamedColumns = {}
    renamedRows = {}
    for feature in features:
        renamedColumns[feature] = "Predicted " + feature
        renamedRows[feature] = "Actual " + feature
    matrix.rename(columns=renamedColumns, index=renamedRows, inplace=True)
    
    # Add Total column and Total index
    matrix["Total"] = pd.Series([0 for i in range(len(features))], index=matrix.index)
    matrix.loc["Total"] = [0 for i in range(len(matrix.columns))]
    
    # Sum the Total values
    for feature in features:
        matrix.loc["Total", "Predicted " + feature] = matrix["Predicted " + feature].sum()
        
    for i in matrix.index:
        matrix.loc[i, "Total"] = matrix.loc[i].sum()
    
    return matrix

def getPrecisionsAndRecalls(confusionMatrix, features):
    ''' '''
    features.sort()
    precisions = {}
    recalls = {}
    
    for feature in features:
        index = "Actual " + feature
        column = "Predicted " + feature
        
        # Handle the case where the classfier doesn't classify any data to a particular label
        precision = 0
        if (confusionMatrix.loc["Total", column] > 0):
            precision = confusionMatrix.loc[index, column] / confusionMatrix.loc["Total", column]

        precisions[feature] = precision
        recall = confusionMatrix.loc[index, column] / confusionMatrix.loc[index, "Total"]
        recalls[feature] = recall
        
    return precisions, recalls

def kFoldSample(k, dataFrame):
    ''' '''
    subsetSize = int(len(dataFrame) / k)
    subsetSizes = (subsetSize, subsetSize + len(dataFrame) % k)    
    samples = []

    for i in range(k):
        randomSample = None
        if (i < k - 1):
            randomSample = dataFrame.sample(n=subsetSizes[0], replace=False)
        else:
            randomSample = dataFrame.sample(n=subsetSizes[1], replace=False)
        dataFrame = dataFrame.drop(randomSample.index)
        samples.append(randomSample)
    return samples

def kFoldCrossValidation(k, dataFrame):
    ''' '''
    samples = kFoldSample(k, dataFrame)
    trainings = []
    tests = []

    # Each sample in the iteration is used as the test set. Get the training set by 
    # taking the whole data frame minuses the data in the test set
    for sample in samples:
        trainings.append(dataFrame.drop(sample.index))
        tests.append(sample)
    return trainings, tests

# def kFoldCrossValidationResult(k, dataFrame, model):
#     ''' '''
#     assert k >= 2, "k must be at least 2"
#     errors = []
#     kTrains, kTests = kFoldCrossValidation(k, dataFrame)

#     for training, test in zip(kTrains, kTests):
#         model.train(training)
#         pred = model.classify(test)
#         errors.append(computeError(pred, test[model.targetFeature]))

#     return sum(errors) / len(errors)

def computeFScores(precisions, recalls):
    ''' '''
    assert len(precisions) == len(recalls), "Length of precisions and recalls must match"
    assert set(precisions.keys()) == set(recalls.keys()), "Precisions and recalls must have the same class labels"
    fScores = {}

    for label in precisions.keys():
        p = precisions[label]
        r = recalls[label]

        # Don't include in the f score computation if p+r is 0
        if (p + r > 0):
            fScore = (2 * p * r) / (p + r)
            fScores[label] = fScore

    return sum(fScores.values()) / len(fScores)
