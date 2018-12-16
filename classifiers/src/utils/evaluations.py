import math
import pandas as pd

def computeError(predictions, actuals):
    ''' Compute the % of misclassification of the prediction '''
    assert len(predictions) == len(actuals), "Number of predictions and actuals must match"
    assert type(predictions) == type(actuals), "Type of predictions and actuals must match"
    misClassified = 0
    for i in predictions.index:
        if (predictions[i] != actuals[i]):
            misClassified += 1
    return misClassified / len(actuals)

def buildConfusionMatrix(predictions, actuals, labels):
    ''' Build the confusion matrix '''
    assert len(predictions) == len(actuals), "Number of predictions and actuals must match"
    assert type(predictions) == type(actuals), "Type of predictions and actuals must match"
    table = {}
    labels.sort()
    
    # Initialize the table with column header and cell values with 0
    for label in labels:
        table["Predicted " + label] = [0 for i in range(len(labels))]
    
    matrix = pd.DataFrame(data=table, index=list(map(lambda x: "Actual " + x, labels)))
    
    # Count the misclassifications
    for i in predictions.index:
        if (predictions[i][0] == actuals[i]):
            matrix.loc["Actual " + predictions[i], "Predicted " + predictions[i]] += 1
        else:
            matrix.loc["Actual " + actuals[i], "Predicted " + predictions[i]] += 1
    
    # Add Total column and Total index
    matrix["Total"] = pd.Series([0 for i in range(len(labels))], index=matrix.index)
    matrix.loc["Total"] = [0 for i in range(len(matrix.columns))]
    
    # Sum the Total values
    for label in labels:
        matrix.loc["Total", "Predicted " + label] = matrix["Predicted " + label].sum()
        
    for i in matrix.index:
        matrix.loc[i, "Total"] = matrix.loc[i].sum()
    
    return matrix

def getPrecisionsAndRecalls(confusionMatrix, labels):
    ''' Get precision and recall values '''
    precisions = {}
    recalls = {}
    
    for label in labels:
        index = "Actual " + label
        column = "Predicted " + label
        
        # Handle the case where the classfier doesn't classify any data to a particular label
        precision = 0
        if (confusionMatrix.loc["Total", column] > 0):
            precision = confusionMatrix.loc[index, column] / confusionMatrix.loc["Total", column]
        precisions[label] = precision

        recall = confusionMatrix.loc[index, column] / confusionMatrix.loc[index, "Total"]
        recalls[label] = recall
        
    return precisions, recalls

def getSensitivityAndSpecifiicy(confusionMatrix, labels):
    ''' Get sensitivity and specificity values '''
    sensitivies = {}
    specificities = {}

    for label in labels:
        index = "Actual " + label
        column = "Predicted " + label

        sensitivity = confusionMatrix.loc[index, column] / confusionMatrix.loc[index, "Total"]
        specificity = 0
        sumOtherLabels = 0
        for otherLabel in labels:
            if (otherLabel != label):
                specificity += confusionMatrix.loc["Actual " + otherLabel, "Predicted " + otherLabel]
                sumOtherLabels += confusionMatrix.loc["Actual " + otherLabel, "Total"]
        specificity /= sumOtherLabels

        sensitivies[label] = sensitivity
        specificities[label] = specificity

    return sensitivies, specificities     


def computeFScores(precisions, recalls):
    ''' Compute the F scores given the precisions and recalls '''
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
        else:
            fScores[label] = 0

    return sum(fScores.values()) / len(fScores), fScores


def tTest(errors1, errors2, k):
    ''' Calculate the t-value of the errors of 2 models '''
    meanErr1 = sum(errors1) / len(errors1)
    meanErr2 = sum(errors2) / len(errors2)
    var = 0

    for e1, e2 in zip(errors1, errors2):
        var += (e1 - e2 - (meanErr1 - meanErr2))**2 / k
    return (meanErr1 - meanErr2) / math.sqrt(var / k)
