import math
import pandas as pd
import numpy as np
import random
import utils.evaluations as ev
from models.supervised import KNearestNeighbors

def splitData(targetFeature, dataFrame, trainingRatio, containAllLabels=True):
    ''' Split data into training and test set '''

    labelValues = dataFrame[targetFeature].unique()
    trainingSize = math.floor(len(dataFrame) * trainingRatio)

    # Ensure that the training and test contain all label values
    training = dataFrame.sample(n=trainingSize, replace=False)
    test = dataFrame.drop(training.index)
    containAllLabels = (not containAllLabels) # Negate this value so it can be used as the while loop terminating condition
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


#rule = {"Sum": (c1, c2, c3, cT), "complement": (c1, c2, total)}
def _generateAddendsRule(total, min, *args):
    ''' @INCOMPLETE '''
    result = {}
    tempMaxRange = total
    tempLocalTotal = 0
    
    for i in range(len(args)):
        value = 0 if (i == len(args) - 1) else randrange(min, tempMaxRange+1)
        key = args[i]
        result[key] = total - (tempLocalTotal + value)
        tempMaxRange = value
        tempLocalTotal += result[key]

    # Sanity check
    generatedTotal = sum(result.values())
    if (total != generatedTotal):
        raise Exception("Total={0} != generated total={1}".format(total, generatedTotal))

    return result

def _complementRule(total, minimum):
    ''' '''
    randomValue = randrange(minimum, total+1)
    return randomValue, total-randomValue


def smote(dataFrame, minorityLabels, targetLabel, kValue):
    ''' Synthetic Minority Oversampling Technique (SMOTE)
        @INCOMPLETE
    '''
    # Treat minority samples as Test data and the remaining as Training data
    minorityDataFrame = dataFrame[dataFrame[targetLabel].isin(minorityLabels)]
    training = dataFrame.drop(index=minorityDataFrame.index)
    knn = KNearestNeighbors(targetLabel, k=kValue)
    knn.train(training)
    continousFeatures = knn.continuousFeatures

    # Initialize the synthetic data frame object with column values
    syntheticData = {targetLabel: []}
    for cf in continousFeatures:
        syntheticData[cf] = []

    for index, row in minorityDataFrame.iterrows():
        # Get nearest neighbors
        nn = knn.getNN(row)

        for neighborIndex in nn:
            neighbor = dataFrame.iloc[neighborIndex]

            # Assign target label to the new data point
            syntheticData[targetLabel].append(row[targetLabel])
            for cf in continousFeatures:
                minValue = min(neighbor[cf], row[cf])
                maxValue = max(neighbor[cf], row[cf])
                value = (neighbor[cf] + row[cf]) / 2
                syntheticData[cf].append(value)
                # Generate new synthetic values by interpolating 
                # between minValue & maxValue for every continous features
                #randomValue = random.uniform(minValue, maxValue)
                # syntheticData[cf].append(randomValue)

    return dataFrame.append(pd.DataFrame(syntheticData), ignore_index=True)
    

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
                error = ev.computeError(kPred["Prediction"], kTest[targetFeature])
                result[k-2].append(error)
    except ValueError as ex:
        print(str(ex) + ". Return early...")

    return result
