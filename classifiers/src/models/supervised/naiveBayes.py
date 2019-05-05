import math
import numpy as np
import pandas as pd
import utils.decorators as decor
from ..model import SupervisedModel, FeatureType

class NaiveBayes(SupervisedModel):
    ''' Naive Bayes classifiier. It takes a target feature as the predicted feature and
        a list of known labels. This classifier stores the conditional probabilities for
        categorical features and mean & standard deviation for continuous features, 
        in a data frame object each.
    '''

    def __init__(self, targetFeature: str, allLabels: [str]):
        ''' Constructor '''
        super().__init__(targetFeature)
        self._labelProbabilities = pd.Series()
        self._allLabels = allLabels

        # {"label1": {"feature1": {value1: probability1}}}
        self._categoricalProbs = dict((label, {}) for label in self._allLabels)

        # {"label1": {"feature1": {"mean": value, "std": value}}}
        self._continuousMeanStd = dict((label, {}) for label in self._allLabels)

    @property
    def name(self) -> str:
        return "Naive Bayes"

    def clear(self):
        ''' Clear the current state and all data of the model.
            This doesn't clear the properties of the model, however.
        '''
        self._labelProbabilities.drop(self._labelProbabilities.index, axis=0, inplace=True)
        
        for features in self._categoricalProbs.values():
            features.clear()
        
        for features in self._continuousMeanStd.values():
            features.clear()

    @decor.elapsedTime
    def train(self, dataFrame, **kwargs):
        ''' Train the naive bayes with the given data frame input '''
        self.clear()
        self._computeLabelProbabilities(dataFrame)
        self._computeConditionalProbabilities(dataFrame)

    @decor.elapsedTime
    def classify(self, dataFrame, **kwargs):
        ''' Classify the input data frame and return a data frame with 2 columns: Prediction and Probability.
            Prediction column denotes the predicted label of a data point and Probability column denotes the
            probability that the prediction is drawn from.
        '''
        super().classify(dataFrame, **kwargs)
        predictions = []
        probabilities = []

        for index, row in dataFrame.iterrows():
            predictProbabilities = []
            
            for label, labelProbability in self._labelProbabilities.items():
                # Transform probability to log space to deal with underflow problem
                logProbability = math.log(labelProbability)

                for feature in dataFrame.columns.values:
                    featureType = super()._getFeatureType(dataFrame, feature)
                    if (featureType == FeatureType.Categorical):
                        # Ignore unseen value and continue to use other probabilities
                        value = row[feature]
                        if (value in self._categoricalProbs[label][feature]):
                            probability = self._categoricalProbs[label][feature][value]
                            logProbability += math.log(probability)
                    
                    elif (featureType == FeatureType.Continuous):
                        meansAndStds = self._continuousMeanStd[label][feature]
                        if ("mean" in meansAndStds and "std" in meansAndStds):
                            mean = meansAndStds["mean"]
                            std = meansAndStds["std"]

                            # Ignore any 0 or near-0 probability. Not worth considering
                            gaussianProb = self._getGaussianProbability(row[feature], mean, std)
                            if (gaussianProb > 0):
                                logProbability += math.log(gaussianProb)

                predictProbabilities.append((label, math.exp(logProbability)))

            # Find the best label and the probability associated to it
            bestLabel, bestProbability = max(predictProbabilities, key=lambda x: x[1])
            predictions.append(bestLabel)
            probabilities.append(bestProbability)

        return self._createResultDataFrame(predictions, probabilities, dataFrame.index)

    def _computeLabelProbabilities(self, dataFrame):
        ''' Compute the label probabilities from the given data frame -> P(C)'''
        labelCounts = dataFrame[self._targetFeature].value_counts()
        for i in labelCounts.index:
            self._labelProbabilities.loc[i] = labelCounts[i] / len(dataFrame)

    def _computeConditionalProbabilities(self, dataFrame):
        ''' Compute all the conditional probabilities of the given data frame -> P(X|C) '''
        features = dataFrame.loc[:, dataFrame.columns != self._targetFeature].columns.values
        featureValueMappings = self._getCategoricalFeatureMappings(dataFrame)

        for label in self._allLabels:
            labelDataFrame = dataFrame[dataFrame[self._targetFeature] == label]
            for feature in features:
                featureType = super()._getFeatureType(dataFrame, feature)

                if (featureType == FeatureType.Categorical):
                    self._categoricalProbs[label][feature] = {}

                    # Set up varibles in preparing to do Laplacian Correction if need to
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
                            probability = 1 / totalSize
                            self._categoricalProbs[label][feature][value] = probability

                    # Compute conditional probabilities of seen values
                    valueCounts = labelDataFrame[feature].value_counts()
                    for value in labelDataFrame[feature].values:
                        probability = (valueCounts[value] + laplacianValue) / totalSize
                        self._categoricalProbs[label][feature][value] = probability

                elif (featureType == FeatureType.Continuous):
                    self._continuousMeanStd[label][feature] = {}
                    mean = labelDataFrame[feature].mean()
                    std = labelDataFrame[feature].std()

                    # Skip zero standard deviation feature. NaN values in continuous table means 
                    # we don't include it when we compute the final probabilities.
                    if (std != 0.0):
                        self._continuousMeanStd[label][feature]["mean"] = mean
                        self._continuousMeanStd[label][feature]["std"] = std
        
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
        for label, features in self._categoricalProbs.items():
            for feature, featureData in features.items():
                for value, probability in featureData.items():
                    if (probability > 1) or (probability < 0):
                        raise ValueError("Probability needs to be between 0 and 1. Row {0} - Column {1}".format(i, j))
