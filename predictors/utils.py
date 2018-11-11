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
    assert len(predictions) == len(actuals), "Number of predictions and actuals must match"
    assert type(predictions) == type(actuals), "Type of predictions and actuals must match"
    misClassified = 0
    for i in predictions.index:
        if (predictions[i] != actuals[i]):
            misClassified += 1
    return 1 - ((len(actuals) - misClassified) / len(actuals))

def buildConfusionMatrix(predictions, actuals, features):
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
    features.sort()
    precisions = {}
    recalls = {}
    
    for feature in features:
        index = "Actual " + feature
        column = "Predicted " + feature
        
        precision = confusionMatrix.loc[index, column] / confusionMatrix.loc["Total", column]
        precisions[feature] = precision
        recall = confusionMatrix.loc[index, column] / confusionMatrix.loc[index, "Total"]
        recalls[feature] = recall
        
    return precisions, recalls

def kFoldSample(k, dataFrame):
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

@decor.elapsedTime
def kFoldCrossValidation(k, dataFrame, model):
    samples = kFoldSample(k, dataFrame)
    accuracies = []
    
    for i in range(len(samples)):
        test = samples[i]
        training = dataFrame.drop(test.index)
        model.train(training)
        predictions = model.classify(test)
        accuracies.append(1 - computeError(predictions, test[model.targetFeature]))
    return accuracies