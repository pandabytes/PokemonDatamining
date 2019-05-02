import numpy as np
import pandas as pd
import utils.decorators as decor
from ..model import SupervisedModel, FeatureType

class KNearestNeighbors(SupervisedModel):
	''' '''

	DistanceMetrics = ["euclidean", "manhattan", "chebyshev"]

	def __init__(self, targetFeature, k=5, distanceMetric="euclidean"):
		''' Constructor '''
		if (distanceMetric not in KNearestNeighbors.DistanceMetrics):
			raise ValueError("Distance metric \"{0}\" is not supported.".format(distanceMetric))
		elif (k <= 0):
			raise ValueError("K value must be greater than 0")

		super().__init__(targetFeature)
		self._k = k
		self._distanceMetric = distanceMetric
		self._training = None
		self._continuousFeatures = []

	@property
	def kValue(self):
		''' Get k value '''
		return self._k

	@kValue.setter
	def kValue(self, value):
		''' Set k value '''
		if (value <= 0):
			raise ValueError("K value must be greater than 0")
		self._k = value

	@property
	def distanceMetric(self):
		''' Get distance metric '''
		return self._distanceMetric

	@distanceMetric.setter
	def distanceMetric(self, value):
		''' Set distance metric '''
		if (value not in KNearestNeighbors.DistanceMetrics):
			raise ValueError("Distance metric \"{0}\" is not supported.".format(value))
		self._distanceMetric = value

	@property
	def continuousFeatures(self):
		''' Get continous features '''
		return self._continuousFeatures

	def train(self, dataFrame, **kwargs):
		''' Train this classifier. Set the training class variable to data
			frame that contains only the continuous features of the training set
		'''
		del self._continuousFeatures[:]

		# Get all continous features
		for feature in dataFrame.columns.values:
			featureType = self._getFeatureType(dataFrame, feature)
			if (featureType == FeatureType.Continuous):
				self._continuousFeatures.append(feature)

		self._training = dataFrame[self._continuousFeatures + [self._targetFeature]]
		
	@decor.elapsedTime
	def classify(self, dataFrame, **kwargs):
		''' '''
		super().classify(dataFrame, **kwargs)
		labelsDistances = {}
		predictions = []
		probabilities = []
		testDataFrame = dataFrame[self._continuousFeatures]

		for i in testDataFrame.index:
			labelsDistances[i] = []

			# Compute the distance from each test point to each training point
			for j in self._training.index:
				if (i != j):
					distance = self._getDistanceMetricFunc()(testDataFrame.loc[i], self._training.loc[j])
					label = self._training.loc[j, self._targetFeature]
					labelsDistances[i].append((label, distance))
				
			# Sort by ascending order and pick the first k points
			labelsDistances[i].sort(key=lambda x: x[1])
			nearestK = labelsDistances[i][:self._k]

			# Count the label values and pick the most frequent one
			labelCounts = KNearestNeighbors._countLabelValues(nearestK)
			prediction, count = max(labelCounts.items(), key=lambda x: x[1])
			probability = count / self._k

			predictions.append(prediction)
			probabilities.append(probability)
			
		return pd.DataFrame({"Prediction": predictions, "Probability": probabilities}, index=dataFrame.index)

	def getNN(self, row):
		''' '''
		row = row[self._continuousFeatures]
		rowDistances = []

		# Compute the distance from each test point to each training point
		for i in self._training.index:
			distance = KNearestNeighbors._euclideanDistance(row, self._training.loc[i])
			rowDistances.append((i, distance))

		# Sort by distances in ascending order
		rowDistances.sort(key=lambda x: x[1])

		return [r[0] for r in rowDistances[:self._k]]

		
	@staticmethod
	def _countLabelValues(labelsDistances):
		''' '''
		labelCounts = {}
		for label, distance in labelsDistances:
			if (label not in labelCounts):
				labelCounts[label] = 1
			else:
				labelCounts[label] += 1
		return labelCounts

	def _getDistanceMetricFunc(self):
		''' '''
		if (self._distanceMetric == "euclidean"):
			return KNearestNeighbors._euclideanDistance
		elif (self._distanceMetric == "manhattan"):
			return KNearestNeighbors._manhattanDistance
		elif (self._distanceMetric == "chebyshev"):
			return KNearestNeighbors._chebyshevDistance

	@staticmethod
	def _euclideanDistance(row1, row2):
		''' Compute the Euclidian distance of two data points (or rows).
		    Assume all the values are continous.
		'''
		distance = 0
		for v1, v2 in zip(row1.values, row2.values):
			distance += (v1 - v2)**2
		return np.sqrt(distance)

	@staticmethod
	def _manhattanDistance(row1, row2):
		''' Compute the Manhattan distance of two data points (or rows).
		    Assume all the values are continous.
		'''
		distance = 0
		for v1, v2 in zip(row1.values, row2.values):
			distance += abs(v1 - v2)
		return distance

	@staticmethod
	def _chebyshevDistance(row1, row2):
		''' Compute the Chebyshev distance of two data points (or rows).
		    Assume all the values are continous.
		'''
		distances = []
		for v1, v2 in zip(row1.values, row2.values):
			distances.append(abs(v1 - v2))
		return max(distances)

