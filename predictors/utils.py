import math
import pandas as pd
import decorators as decor
import numpy as np
import matplotlib.pyplot as plt

def splitData(targetFeature, dataFrame, trainingRatio):
    ''' Split data into training and test set '''

    labelValues = dataFrame[targetFeature].unique()
    trainingSize = math.floor(len(dataFrame) * trainingRatio)

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
    ''' Inject fake minority samples to the data.
        @UNUSED and @INCOMPLETE
    '''

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
            print("k =", k)
            for kTrain, kTest in zip(kTrainings, kTests):
                model.train(kTrain, quiet=True)
                kPred = model.classify(kTest.drop([targetFeature], axis=1), quiet=True)
                error = computeError(kPred["Prediction"], kTest[targetFeature])
                result[k-2].append(error)
    except ValueError as ex:
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

def generateRandomThresholds(n, labels):
    ''' Generate random thresholds for the given labels '''
    thresholds = []
    for i in range(n):
        randomProbs = np.random.dirichlet(np.ones(len(labels)), size=1)[0]
        series = pd.Series(randomProbs, index=labels)
        thresholds.append(series)
    return thresholds

def generateThresholds(minThreshold, maxThreshold, numThreshold, sizePerLabel, labels):
    ''' Generate thresholds with the given interval '''
    if not (0.0 < minThreshold < 1.0 or 0.0 < maxThreshold < 1.0):
        raise ValueError("Min or max threshold must be between 0 and 1 exclusively")

    thresholds = []
    labelsSet = set(labels)
    thresholdValues = np.linspace(minThreshold, maxThreshold, num=numThreshold)
    labelSizes = dict([(l, 0) for l in labels])

    for i in thresholdValues:
        for j in thresholdValues:
            a = i
            b = j
            c = 1 - (i + j)
            #print("{0:.2f} + {1:.2f} + {2:.2f} = {3}".format(a,b,c, a+b+c))
            for label in labels:
                if (labelSizes[label] >= sizePerLabel):
                    continue
                labelSizes[label] += 1
                otherLabels = sorted(list(labelsSet - set([label])))
                series = pd.Series({label: a, otherLabels[0]: b, otherLabels[1]: c})
                thresholds.append(series)

            # if (c != 0) and (a + b < 1):
            #     for label in labels:
            #         otherLabels = sorted(list(labelsSet - set([label])))
            #         series = pd.Series({label: a, otherLabels[0]: b, otherLabels[1]: c})
            #         thresholds.append(series)
        # for label in labels:
        #     threshold = i
        #     otherThreshold1 = np.random.uniform(0.0, 1-threshold)
        #     otherThreshold2 = 1 - (threshold + otherThreshold1)

        #     otherLabels = list(labelsSet - set([label]))
        #     series = pd.Series({label: threshold, otherLabels[0]: otherThreshold1, otherLabels[1]: otherThreshold2})
        #     thresholds.append(series)

    return thresholds

def tTest(errors1, errors2, k):
    ''' Calculate the t-value of the errors of 2 models '''
    meanErr1 = sum(errors1) / len(errors1)
    meanErr2 = sum(errors2) / len(errors2)
    var = 0

    for e1, e2 in zip(errors1, errors2):
        var += (e1 - e2 - (meanErr1 - meanErr2))**2 / k
    return (meanErr1 - meanErr2) / math.sqrt(var / k)

def precisionRecallCurve(actuals, predictionScores):
    ''' Get the precision and recall values to plot the Precision-Recall Curve.
        This function treat each label as "positive" and others as "negative".
        In other words, this function "normalizes" the the prediction labels to be
        binary label. 

        Note: This is different from the getPrecisionsAndRecalls() function above where the
        labels are transformed to binary labels.
    '''
    # Use the prediction score as the threshold
    thresholds = set(predictionScores)
    precisions = {}
    recalls = {}

    for label in actuals:
        labelPrecisions = []
        labelRecalls = []
        binarizedLabels = []

        # Tranfrom the labels to binary form: 0 and 1
        for i, j in zip(actuals, predictionScores):
            binarizedLabel = int(i == label)
            binarizedLabels.append((binarizedLabel, j))

        # Compute precision and recall for each threshold
        for t in thresholds:
            positive = 0
            truePositive = 0
            truePositiveFalsePositive = 0

            for b in binarizedLabels:
                if (b[0] == 1):
                    positive += 1
                if (b[0] == 1 and b[1] >= t):
                    truePositive += 1
                if (b[1] >= t):
                    truePositiveFalsePositive += 1

            precision = truePositive / truePositiveFalsePositive
            recall = truePositive / positive
            labelPrecisions.append(precision)
            labelRecalls.append(recall)

        precisions[label] = labelPrecisions
        recalls[label] = labelRecalls

    return precisions, recalls

def plotPrecisionRecallCurve(precisions, recalls, labelColors):
    ''' Plot the Precision-Recall Curve for each label '''
    if (len(precisions) != len(recalls)):
        raise ValueError("Length of precisions and recalls must match")
    if (precisions.keys() != recalls.keys()):
        raise ValueError("Keys in precisions and recalls must match")

    for label in precisions.keys():
        labelPrecisions = precisions[label]
        labelRecalls = recalls[label]

        # Group precision and recall in tuple and sort by recall in ascending order
        labelPr = [(1, 0)]
        for i, j in zip(labelPrecisions, labelRecalls):
            labelPr.append((i, j))
        labelPr.sort(key=lambda x: x[1])
        
        # Plot the PR curve
        plt.plot([i[1] for i in labelPr], [i[0] for i in labelPr], c=labelColors[label], label=label)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="best")
    plt.show()

