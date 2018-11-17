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

	def train(self, dataFrame):
		''' '''
		return self._getConditionalProbabilities(dataFrame)
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

	def _getConditionalProbabilities(self, dataFrame):
		''' '''
		features = dataFrame.loc[:, dataFrame.columns != self._targetFeature].columns.values
		labelProbabilities = self._getLabelProbabilities(dataFrame)
		labelCounts = dataFrame[self._targetFeature].value_counts()
		#table = pd.DataFrame(index=labelCounts.index)
		self._categoricalProbTable = pd.DataFrame(index=labelCounts.index)
		self._continuousProbTable = pd.DataFrame(index=labelCounts.index)

		# Need to handle zero probability
		# Need to handle zero standard deviation
		# Need to see all possible feature values of each feature
		for label in labelCounts.index:
			labelDataFrame = dataFrame[dataFrame[self._targetFeature] == label]
			for feature in features:
				featureType = super()._getFeatureType(dataFrame, feature)
				if (featureType == "categorical"):
					valueCounts = labelDataFrame[feature].value_counts()
					for value in valueCounts.index:
						probability = valueCounts[value] / len(labelDataFrame)
						columnName = "{0}={1}".format(feature, value)
						self._categoricalProbTable.loc[label, columnName] = probability
				elif (featureType == "continuous"):
					mean = labelDataFrame[feature].mean()
					std = labelDataFrame[feature].std()
					
					# Skip zero standard deviation feature
					if (std != 0.0):
						#print("Saving mean & std of feature {0} -- label {1} -- mean {2} -- std {3}".format(feature, label, mean, std))
						columnMean = "{0}=mean".format(feature)
						columnStd = "{0}=std".format(feature)
						self._continuousProbTable.loc[label, columnMean] = mean
						self._continuousProbTable.loc[label, columnStd] = std
					else: print("****Skipping {0} feature for label {1}. Standard deviation = 0".format(feature, label))
		


	@staticmethod
	def ComputeGaussianProbability(value, mean, std):
		''' '''
		return (1 / (math.sqrt(2 * (std**2) * math.pi))) * math.exp(-(value - mean)**2 / (2 * std**2))
				

		