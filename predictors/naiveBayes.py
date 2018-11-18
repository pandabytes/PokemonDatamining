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
		#labelProbabilities = self._getLabelProbabilities(dataFrame)
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
					
					#laplacianValue = 1 if (len(labelDataFrame) == 0) else 0

					valueCounts = labelDataFrame[feature].value_counts()
					for value in valueCounts.index:
						probability = valueCounts[value] / len(labelDataFrame)
						columnName = "{0}={1}".format(feature, value)
						self._categoricalProbTable.loc[label, columnName] = probability



				elif (featureType == "continuous"):
					mean = labelDataFrame[feature].mean()
					std = labelDataFrame[feature].std()
					
					# Skip zero standard deviation feature
					# NaN values in continuous table means we don't include it when we compute the final probabilities
					if (std != 0.0):
						columnMean = "{0}=mean".format(feature)
						columnStd = "{0}=std".format(feature)
						self._continuousProbTable.loc[label, columnMean] = mean
						self._continuousProbTable.loc[label, columnStd] = std
					else: print("****Skipping {0} feature for label {1}. Standard deviation = 0".format(feature, label))
		


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

				

		