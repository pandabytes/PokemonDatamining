import math
import pandas as pd
import decorators as decor
import numpy as np

def splitData(targetFeature, dataFrame, trainingRatio):
    ''' Split data into training and test set '''

    labelValues = dataFrame[targetFeature].unique()
    trainingSize = math.floor(len(dataFrame) * trainingRatio)
    testSize = len(dataFrame) - trainingSize

    # Ensure that the training and test contain all label values
    training = dataFrame.sample(n=trainingSize, replace=False, random_state=np.random.RandomState())
    test = dataFrame.drop(training.index)
    containAllLabels = False
    while (not containAllLabels):
        for labelValue in labelValues:
            if (labelValue not in training[targetFeature].unique() or labelValue not in test[targetFeature].unique()):
                containAllLabels = False
                training = dataFrame.sample(n=trainingSize, replace=False)
                break
            else:
                containAllLabels = True
                test = dataFrame.drop(training.index)

    return training, test

def injectMinoritySample(minLabels, target, dataFrame):
    ''' Inject fake minority samples to the data '''

    total = len(dataFrame) 
    maxSizeLabel = dataFrame[target].value_counts().max()
    ratio = dataFrame[target].value_counts().max() / float(total)
    sizePerLabel = int(len(dataFrame) / float(len(dataFrame[target].unique())))
    result = dataFrame

    for label in minLabels:
        labelDataFrame = result[result[target] == label]
        labelSize = len(labelDataFrame)

        #while (len(labelDataFrame) < sizePerLabel):
        while (len(labelDataFrame) < maxSizeLabel):
            result = result.append(labelDataFrame.sample(labelSize), ignore_index=True)
            labelDataFrame = result[result[target] == label]
        print(label, len(labelDataFrame))

    return result

def computeError(predictions, actuals):
    ''' Compute the % of misclassification of the prediction '''
    assert len(predictions) == len(actuals), "Number of predictions and actuals must match"
    assert type(predictions) == type(actuals), "Type of predictions and actuals must match"
    misClassified = 0
    for i in predictions.index:
        if (predictions[i] != actuals[i]):
            misClassified += 1
    return 1 - ((len(actuals) - misClassified) / len(actuals))

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

def kFoldSample(k, dataFrame):
    ''' Parition the data into k fold samples '''
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

def kFoldStratifiedSample(k, dataFrame, targetFeature):
    ''' Partition the given data into k stratified sample '''
    labelMinSize = min(dataFrame[targetFeature].value_counts())
    if (k > labelMinSize):
        raise ValueError("k exceeds the label with the smallest number of data points. " + \
                         "k={0}, smallest # of data points of a label={1}".format(k, labelMinSize))
        
    samples = []
    labelCounts = dataFrame[targetFeature].value_counts()

    # Compute the size of each label for each k fold
    labelSampleSizes = {}
    for label in labelCounts.index:
        labelSampleSizes[label] = math.floor(labelCounts[label] / k)

    # Run through each k-fold to partition the data
    for kIter in range(k):
        randomSample = pd.DataFrame()
        
        # If it's the last iteration then simply include all the remaining data
        if (kIter == k - 1):
            randomSample = pd.concat([randomSample, dataFrame])
        else:
            # Get a small portion from each label by taking the total number of 
            # data belonging to a label and divide it by k
            for label in labelCounts.index:
                labelDataFrame = dataFrame[dataFrame[targetFeature] == label]
                labelSample = labelDataFrame.sample(n=labelSampleSizes[label], replace=False)

                # Aggregate to the current sample
                randomSample = pd.concat([randomSample, labelSample])
                dataFrame = dataFrame.drop(labelSample.index)

        samples.append(randomSample)
    return samples

def kFoldCrossValidation(k, dataFrame, stratified=False, targetFeature=None):
    ''' Return 2 lists of training and test of k cross validation '''
    assert k > 1, "k value must be at least 2"
    if (stratified):
        if (targetFeature != None):
            samples = kFoldStratifiedSample(k, dataFrame, targetFeature)
        else:
            raise ValueError("Target feature must be supplemented when stratified sample is specified")
    else:
        samples = kFoldSample(k, dataFrame)

    trainings = []
    tests = []

    # Each sample in the iteration is used as the test set. Get the training set by 
    # taking the whole data frame minuses the data in the test set
    for sample in samples:
        trainings.append(dataFrame.drop(sample.index))
        tests.append(sample)
    return trainings, tests

def kFoldCrossValidationResult(kFolds, targetFeature, dataFrame, model):
    ''' Given k, perform k cross validation on the given model '''
    assert kFolds >= 2, "kFolds must be at least 2"
    result = []

    try:
        for k in range(2, kFolds + 1):
            kTrainings, kTests = kFoldCrossValidation(k, dataFrame, True, targetFeature)
            result.append([])
            print("k = ", k)
            for kTrain, kTest in zip(kTrainings, kTests):
                model.train(kTrain, quiet=True)
                kPred = model.classify(kTest.drop([targetFeature], axis=1), quiet=True)
                error = computeError(kPred, kTest["Group"])
                result[k-2].append(error)
    except ValueError as ex:
        #print("Warning: k={0} exceeds {1}, which is the minimum # of data points a label. Return early...".format(k))
        print(str(ex) + ". Return early...")

    return result

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