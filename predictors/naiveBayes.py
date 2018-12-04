import math
import numpy as np
import pandas as pd
import decorators as decor
from model import SupervisedModel, FeatureType

class NaiveBayes(SupervisedModel):
    ''' Naive Bayes classifiier. It takes a target feature as the predicted feature and
        a list of known labels. This classifier stores the conditional probabilities for
        categorical features and mean & standard deviation for continuous features, 
        in a data frame object each.
    '''

    # Static variable
    ColumnNameFormat = "{0}={1}"

    def __init__(self, targetFeature, allLabels, probThresholds=None):
        ''' Constructor '''
        super().__init__(targetFeature, probThresholds)
        self._labelProbabilities = None
        self._categoricalProbTable = None
        self._continuousMeanStdTable = None
        self._allLabels = allLabels

    @decor.elapsedTime
    def train(self, dataFrame, **kwargs):
        ''' Train the naive bayes with the given data frame input '''
        self._computeLabelProbabilities(dataFrame)
        self._computeConditionalProbabilities(dataFrame)

    @decor.elapsedTime
    def classify(self, dataFrame, **kwargs):
        ''' Classify the input data frame and return a data frame with 2 columns: Prediction and Probability.
            Prediction column denotes the predicted label of a data point and Probability column denotes the
            probability that the prediction is drawn from.
        '''
        assert self._targetFeature not in dataFrame.columns.values, "Test data must not contain the target feature \"%s\"" % self._targetFeature
        predictions = []
        probabilities = []
        indices = []

        for index, row in dataFrame.iterrows():
            predictProbabilities = []
            
            for label, labelProbability in self._labelProbabilities.items():
                # Transform probability to log space to deal with underflow problem
                logProbability = math.log(labelProbability)

                for feature in dataFrame.columns.values:
                    featureType = super()._getFeatureType(dataFrame, feature)
                    if (featureType == FeatureType.Categorical):
                        columnName = NaiveBayes.ColumnNameFormat.format(feature, row[feature])
                        columnValues = self._categoricalProbTable.columns.values

                        # Ignore unseen value and continue to use other probabilities
                        if (columnName in columnValues):
                            logProbability += math.log(self._categoricalProbTable.loc[label, columnName])
                    
                    elif (featureType == FeatureType.Continuous):
                        columnMean = NaiveBayes.ColumnNameFormat.format(feature, "mean")
                        columnStd = NaiveBayes.ColumnNameFormat.format(feature, "std")
                        columnValues = self._continuousMeanStdTable.columns.values

                        # Ignore trivial feature and continue to use other probabilities
                        if (columnMean in columnValues and columnStd in columnValues):
                            mean = self._continuousMeanStdTable.loc[label, columnMean]
                            std = self._continuousMeanStdTable.loc[label, columnStd]
                            if (not math.isnan(mean) and not math.isnan(std)):
                                # Ignore any 0 or near-0 probability. Not worth considering
                                gaussianProb = self._getGaussianProbability(row[feature], mean, std)
                                if (gaussianProb > 0):
                                    logProbability += math.log(gaussianProb)

                predictProbabilities.append((label,  math.exp(logProbability)))

            # Scale by probability threshold if available
            predictionSeries = pd.Series([p[1] for p in predictProbabilities], index=[p[0] for p in predictProbabilities])
            labelProbs = self._scaleByThresholds(predictionSeries)

            bestLabel, bestProbability = max(labelProbs.items(), key=lambda x: x[1])
            predictions.append(bestLabel)
            probabilities.append(bestProbability)
            indices.append(index)
            
            # Sort by the log probability value stored in tuple
            # Convert the log probability to normal probability
            # bestLabel, bestLogProbability = max(predictProbabilities, key=lambda x: x[1])
            # probability = math.exp(bestLogProbability)
            # predictions.append(bestLabel)
            # probabilities.append(probability)
            # indices.append(index)

        return pd.DataFrame({"Prediction": predictions, "Probability": probabilities}, index=indices)

    def _scaleByThresholds(self, predictionSeries):
        ''' Scale the probability threshold of the model if
            the thresholds are available
        '''
        labelProbs = predictionSeries
        if (self._probThresholds is not None):
            labelProbs *= self._probThresholds
        return labelProbs

    def _computeLabelProbabilities(self, dataFrame):
        ''' Compute the label probabilities from the given data frame -> P(C)'''
        self._labelProbabilities = pd.Series()
        labelCounts = dataFrame[self._targetFeature].value_counts()
        for i in labelCounts.index:
            self._labelProbabilities[i] = labelCounts[i] / len(dataFrame)

    def _computeConditionalProbabilities(self, dataFrame):
        ''' Compute all the conditional probabilities of the given data frame -> P(X|C) '''
        features = dataFrame.loc[:, dataFrame.columns != self._targetFeature].columns.values
        featureValueMappings = self._getCategoricalFeatureMappings(dataFrame)
        self._categoricalProbTable = pd.DataFrame(index=self._allLabels)
        self._continuousMeanStdTable = pd.DataFrame(index=self._allLabels)

        for label in self._allLabels:
            labelDataFrame = dataFrame[dataFrame[self._targetFeature] == label]
            for feature in features:
                featureType = super()._getFeatureType(dataFrame, feature)

                if (featureType == FeatureType.Categorical):
                    
                    laplacianValue = 0
                    totalSize = len(labelDataFrame) 
                    unseenValuesSet = set(featureValueMappings[feature]) - set(labelDataFrame[feature].unique())

                    # If there are unseen values, use Laplacian Correction to compute unseen feature probabilities
                    # Else all feature values are in the training dataset
                    if (len(unseenValuesSet) > 0):
                        # Do Laplacian Correction. Increase the total size by each available value
                        laplacianValue = 1
                        totalSize += len(featureValueMappings[feature])
                        for value in unseenValuesSet:
                            columnName = NaiveBayes.ColumnNameFormat.format(feature, value)
                            probability = 1 / totalSize
                            self._categoricalProbTable.loc[label, columnName] = probability

                    # Compute conditional probabilities of seen values
                    valueCounts = labelDataFrame[feature].value_counts()
                    for value in labelDataFrame[feature].values:
                        columnName = NaiveBayes.ColumnNameFormat.format(feature, value)
                        probability = (valueCounts[value] + laplacianValue) / totalSize
                        self._categoricalProbTable.loc[label, columnName] = probability

                elif (featureType == FeatureType.Continuous):
                    mean = labelDataFrame[feature].mean()
                    std = labelDataFrame[feature].std()
                    
                    # Skip zero standard deviation feature. NaN values in continuous table means 
                    # we don't include it when we compute the final probabilities.
                    if (std != 0.0):
                        columnMean = NaiveBayes.ColumnNameFormat.format(feature, "mean")
                        columnStd = NaiveBayes.ColumnNameFormat.format(feature, "std")
                        self._continuousMeanStdTable.loc[label, columnMean] = mean
                        self._continuousMeanStdTable.loc[label, columnStd] = std
        
        # Verify probability values are approriate
        self._verifyProbabilities()

    def _getGaussianProbability(self, value, mean, std):
        ''' Use Gaussian Distribution to compute the probability '''
        return (1 / (math.sqrt(2 * (std**2) * math.pi))) * math.exp(-(value - mean)**2 / (2 * std**2))

    def _getCategoricalFeatureMappings(self, dataFrame):
        ''' Get all the values of each categorical feature '''
        mappings = {}
        features = dataFrame.loc[:, dataFrame.columns != self._targetFeature].columns.values
        for feature in features:
            featureType = super()._getFeatureType(dataFrame, feature)
            if (featureType == FeatureType.Categorical):
                mappings[feature] = dataFrame[feature].unique()

        return mappings

    def _verifyProbabilities(self):
        ''' Sanity test for verifying if the computed probability values are valid '''
        for i in self._categoricalProbTable.index:
            for j in self._categoricalProbTable.columns.values:
                probability = self._categoricalProbTable.loc[i, j]
                if (probability > 1) or (probability < 0):
                    raise ValueError("Probability needs to be between 0 and 1. Row {0} - Column {1}".format(i, j))


                

