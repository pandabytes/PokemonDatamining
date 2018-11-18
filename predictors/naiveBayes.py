import math
import numpy as np
import pandas as pd
import decorators as decor
from model import SupervisedModel

class NaiveBayes(SupervisedModel):
	def __init__(self, targetFeature):
		''' '''
		super().__init__(targetFeature)
		self._categoricalProbTable = None
		self._continuousProbTable = None

	def train(self, dataFrame, labels):
		''' '''
		self._getConditionalProbabilities(dataFrame, labels)
		# print(labelProbabilities)
		# print(super()._getFeatureType(dataFrame, "Group"))

	# def classify(self, dataFrame):
	# 	''' '''
	# 	pass

	def _getLabelProbabilities(self, dataFrame):
		''' '''
		probabilities = {}
		labelCounts = dataFrame[self._targetFeature].value_counts()
		for i in labelCounts.index:
			probabilities[i] = labelCounts[i] / len(dataFrame)
		return probabilities

	def _getConditionalProbabilities(self, dataFrame, labels):
		''' '''
		features = dataFrame.loc[:, dataFrame.columns != self._targetFeature].columns.values
		featureValueMappings = self._getCategoricalFeatureMappings(dataFrame)
		self._categoricalProbTable = pd.DataFrame(index=labels)
		self._continuousProbTable = pd.DataFrame(index=labels)

		# Need to handle zero probability
		# Need to handle zero standard deviation
		# Need to see all possible feature values of each feature
		# Ignore unseen feature values and use other feature values to compute the posterior probabilities
		for label in labels:
			labelDataFrame = dataFrame[dataFrame[self._targetFeature] == label]
			for feature in features:
				featureType = super()._getFeatureType(dataFrame, feature)
				if (featureType == "categorical"):
					
					laplacianValue = 0
					totalSize = len(labelDataFrame) 
					unseenValuesSet = set(featureValueMappings[feature]) - set(labelDataFrame[feature].unique())

					# If there are unseen values, do Laplacian Correction. Else all feature values are in the training dataset
					if (len(unseenValuesSet) > 0):
						# Do Laplacian Correction. Increase the total size by each available value
						laplacianValue = 1
						totalSize += len(featureValueMappings[feature])
						for value in unseenValuesSet:
							columnName = "{0}={1}".format(feature, value)
							probability = 1 / totalSize
							self._categoricalProbTable.loc[label, columnName] = probability
							#print("Label: {0} --- Unseen feature {1} --- value {2} --- prob 1/{3}".format(label, feature, value, totalSize))

					valueCounts = labelDataFrame[feature].value_counts()
					for value in labelDataFrame[feature].values:
						columnName = "{0}={1}".format(feature, value)
						probability = (valueCounts[value] + laplacianValue) / totalSize
						self._categoricalProbTable.loc[label, columnName] = probability
						#print("*** Label: {0} --- Feature {1} --- value {2} --- prob {3}/{4}".format(label, feature, value, valueCounts[value] + laplacianValue, totalSize))

				elif (featureType == "continuous"):
					mean = labelDataFrame[feature].mean()
					std = labelDataFrame[feature].std()
					
					# Skip zero standard deviation feature
					# NaN values in continuous table means we don't include it when we compute the final probabilities
					if (std != 0.0):
						columnMean = "{0} mean".format(feature)
						columnStd = "{0} std".format(feature)
						self._continuousProbTable.loc[label, columnMean] = mean
						self._continuousProbTable.loc[label, columnStd] = std
					else: print("****Skipping {0} feature for label {1}. Standard deviation = 0".format(feature, label))
		
		self._verifyProbabilities()


	#def _getCategoricalConditionalProbabilities(self, labelDataFrame, )


	@staticmethod
	def ComputeGaussianProbability(value, mean, std):
		''' '''
		return (1 / (math.sqrt(2 * (std**2) * math.pi))) * math.exp(-(value - mean)**2 / (2 * std**2))

	def _getCategoricalFeatureMappings(self, dataFrame):
		''' '''
		mappings = {}
		features = dataFrame.loc[:, dataFrame.columns != self._targetFeature].columns.values
		for feature in features:
			featureType = super()._getFeatureType(dataFrame, feature)
			if (featureType == "categorical"):
				mappings[feature] = dataFrame[feature].unique()

		return mappings

	def _verifyProbabilities(self):
		''' '''
		for i in self._categoricalProbTable.index:
			for j in self._categoricalProbTable.columns.values:
				probability = self._categoricalProbTable.loc[i, j]
				if (probability > 1) or (probability < 0):
					raise ValueError("Probability needs be between 0 and 1. Row {0} - Column {1}".format(i, j))


				

		