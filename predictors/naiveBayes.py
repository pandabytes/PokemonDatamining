import math
import numpy as np
import pandas as pd
import decorators as decor
from model import SupervisedModel

class NaiveBayes(SupervisedModel):
	def __init__(self, targetFeature):
		super().__init__(targetFeature)

	def train(self, dataFrame):
		priorProbabilities = self._getPriorProbabilities(dataFrame)
		return self._getConditionalProbabilities(dataFrame)
		# print(priorProbabilities)
		# print(super()._getFeatureType(dataFrame, "Group"))

	# def classify(self, dataFrame):
	# 	''' '''
	# 	pass

	def _getPriorProbabilities(self, dataFrame):
		''' '''
		probabilities = {}
		labelCounts = dataFrame[self._targetFeature].value_counts()
		for i in labelCounts.index:
			probabilities[i] = labelCounts[i] / len(dataFrame)
		return probabilities

	def _getConditionalProbabilities(self, dataFrame):
		''' '''
		features = dataFrame.loc[:, dataFrame.columns != self._targetFeature].columns.values
		priorProbabilities = self._getPriorProbabilities(dataFrame)
		labelCounts = dataFrame[self._targetFeature].value_counts()
		table = pd.DataFrame(index=labelCounts.index)

		# for feature in features:
		# 	featureType = super()._getFeatureType(dataFrame, feature)
		# 	if (featureType == "categorical"):
		# 		conditionalFeatureProbs = {} #{"Legendary": 0.98, "Ultra": 0.02}}

		# 		for label in labelCounts.index:
		# 			productProb = 1
		# 			labelDataFrame = dataFrame[dataFrame[self._targetFeature] == label]
		# 			valueCounts = labelDataFrame[feature].value_counts()
		# 			for value in valueCounts.index:
		# 				conditionalFeatureProbs[(feature, value)] = valueCounts[value] / len(labelDataFrame)
		# 			conditionalFeatureProbs[label] = productProb


		# 		table[feature] = conditionalFeatureProbs
		# 		# print(conditionalFeatureProbs)
		# 		break

		# Need to handle zero probability
		# Need to handle zero variance
		# Need to see all possible feature values of each feature
		for label in labelCounts.index:
			labelDataFrame = dataFrame[dataFrame[self._targetFeature] == label]
			for feature in features:
				featureType = super()._getFeatureType(dataFrame, feature)
				if (featureType == "categorical"): pass
					# valueCounts = labelDataFrame[feature].value_counts()
					# for value in valueCounts.index:
					# 	probability = valueCounts[value] / len(labelDataFrame)
					# 	columnName = "{0}={1}".format(feature, value)
					# 	table.loc[label, columnName] = probability
				elif (featureType == "continuous"):
					# raise NotImplementedError()
					mean = labelDataFrame[feature].mean()
					std = labelDataFrame[feature].std()
					#print(label, feature, mean, std, 2 * (std**2) * math.pi)
					
					# Skip zero standard deviation feature
					if (std == 0.0):
						continue

					operand1 = (1 / (math.sqrt(2 * (std**2) * math.pi)))
					for value in labelDataFrame[feature]:

						operand2 = math.exp(-(value - mean)**2 / (2 * std**2))
						probability = operand1 * operand2

						columnName = "{0}={1}".format(feature, value)
						table.loc[label, columnName] = probability

		return table.fillna(0.0)
				

		