import numpy as np
import pandas as pd
import decorators as decor
from model import SupervisedModel

class NaiveBayes(SupervisedModel):
	def __init__(self, targetFeature):
		super().__init__(targetFeature)

	def train(self, dataFrame):
		priorProbabilities = self._getPriorProbabilities(dataFrame)
		self._getConditionalProbabilities(dataFrame)
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
		features = dataFrame.columns.values
		priorProbabilities = self._getPriorProbabilities(dataFrame)
		labelCounts = dataFrame[self._targetFeature].value_counts()
		print(labelCounts)
		table = {}

		for feature in features:
			featureType = super()._getFeatureType(dataFrame, feature)
			if (featureType == "categorical"):
				conditionalFeatureProb = {feature: {}}#{"Legendary": 0.98, "Ultra": 0.02}}
				valueCounts = dataFrame[feature].value_counts()

				for label in labelCounts.index:
					productProb = 1
					for value in valueCounts.index:
						productProb *= valueCounts[value] / labelCounts[label]
						print( valueCounts[value], "/", labelCounts[label], "=", valueCounts[value] / labelCounts[label])
					conditionalFeatureProb[feature][label] = productProb

				print(conditionalFeatureProb)
				return		

		