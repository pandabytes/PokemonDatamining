import numpy as np
import pandas as pd
import threading

class TreeNode:
    ''' Base class for representing tree node'''
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

class DecisionNode(TreeNode):
    ''' Class node that contains the split information. It contains the
        the feature is chosen to be split and its value
    '''
    def __init__(self, left=None, right=None, feature=None, featureValue=None):
        super().__init__(left, right)
        self.feature = feature
        self.featureValue = featureValue

class LeafNode(TreeNode):
    ''' Class node that contains the prediction of the sample's label '''
    def __init__(self, prediction):
        super().__init__(None, None)
        self.prediction = prediction

class DecisionTree:
    ''' Decision Tree classifiier. It takes a target feature as the predicted feature.
        It contains a reference to a TreeNode object after it is trained.
        Use Gini Impurity and build a binary tree
    '''
    def __init__(self, targetFeature):
        self._targetFeature = targetFeature
        self._trainedRootNode = None

    @property
    def targetFeature(self):
        return self._targetFeature
    
    @targetFeature.setter
    def data(self, value):
        self._targetFeature = value
        
    def informationGain(self, left, right, currentImpurity):
        ''' Compute the information gain of the split '''
        p = len(left) / float(len(left) + len(right))
        childrenImpurity =  (p * self.giniImpurity(left)) + ((1-p) * self.giniImpurity(right))
        return currentImpurity - childrenImpurity
    
    def giniImpurity(self, dataFrame):
        ''' Compute the Gini Impurity of the given data frame '''
        labelCounts = dataFrame[self._targetFeature].value_counts()
        impurity = 1
        for label in labelCounts.index:
            probability = labelCounts[label] / float(len(dataFrame))
            impurity -= probability**2
        return impurity
    
    def partition(self, dataFrame, feature, value):
        ''' Partition the given data frame into 2 sub-data frames by the given feature and its value '''
        trueData, falseData = None, None
        
        if (dataFrame.dtypes[feature].type == np.int64 or dataFrame.dtypes[feature].type == np.float64):
            assert type(value) == int or type(value) == float, "Numeric feature must be passed with a numeric value"
            trueData, falseData = self.partitionContinuous(dataFrame, feature, value)

        elif (dataFrame.dtypes[feature].type == np.object_):
            assert type(value) == str, "Categorical feature must be passed with a string value"
            trueData, falseData = self.partitionDiscreteBinary(dataFrame, feature, value)

        else:
            raise ValueError("Invalid feature %s type" % dataFrame.dtypes[feature].type)
            
        return trueData, falseData
    
    def partitionContinuous(self, dataFrame, feature, quantileValue):
        ''' Partition continous values with a given feature and quantile value. '''
        trueData = dataFrame[dataFrame[feature] >= quantileValue]
        falseData = dataFrame[dataFrame[feature] < quantileValue]  
        return trueData, falseData
    
    def partitionDiscrete(self, dataFrame, feature):
        ''' Partition a categorical feature into x number of categorical value of the given feature '''
        partitions = []
        for value in dataFrame[feature].unique():
            partitions.append(dataFrame[dataFrame[feature] == value])
        return partitions
                
    def partitionDiscreteBinary(self, dataFrame, feature, value):
        ''' Partition a categorical feature into 2 sub-panda frames '''
        trueData = dataFrame[dataFrame[feature] == value]
        falseData = dataFrame[dataFrame[feature] != value]  
        return trueData, falseData
    
    def findBestFeature(self, dataFrame, quantiles=[0.2, 0.4, 0.6, 0.8]):
        ''' Find the best feature to split the given data frame. Quantiles are optional and 
            are only used for continous features
        '''
        bestGain = 0.0
        currentImpurity = self.giniImpurity(dataFrame)
        bestFeature = None
        bestFeatureValue = None
        features = dataFrame.loc[:, dataFrame.columns != self._targetFeature].columns
        
        for feature in features:
            if (dataFrame.dtypes[feature].type == np.int64 or dataFrame.dtypes[feature].type == np.float64):
                values = pd.Series([i for i in dataFrame[feature]])
                quantileValues = values.quantile(quantiles, "linear")

                # Find the best quantile value
                for quantileValue in quantileValues:
                    trueData, falseData = self.partition(dataFrame, feature, quantileValue)

                    if (len(trueData) == 0 or len(falseData) == 0):
                        continue
                        
                    infoGain = self.informationGain(trueData, falseData, currentImpurity)
                    if (infoGain > bestGain):
                        bestGain = infoGain
                        bestFeature = feature
                        bestFeatureValue = quantileValue

            elif (dataFrame.dtypes[feature].type == np.object_):
                for featureValue in dataFrame[feature].unique():
                    trueData, falseData = self.partition(dataFrame, feature, featureValue)
                    
                    if (len(trueData) == 0 or len(falseData) == 0):
                        continue
                        
                    infoGain = self.informationGain(trueData, falseData, currentImpurity)
                    if (infoGain > bestGain):
                        bestGain = infoGain
                        bestFeature = feature
                        bestFeatureValue = featureValue
            else:
                raise ValueError("Invalid feature %s type" % dataFrame.dtypes[feature].type)
                    
        return bestFeature, bestFeatureValue, bestGain
        
    def train(self, dataFrame):
        ''' Train the decision tree with the given data frame input '''
        self._trainedRootNode = self._buildTree(dataFrame)

    def classify(self, dataFrame):
        ''' Classify the given data frame '''
        result = {}
        for i, row in dataFrame.iterrows():
            result[i] = self._classifyOneSample(row, self._trainedRootNode)
        return pd.Series(result)

    def _classifyOneSample(self, row, node):
        ''' Classfiy one sample '''
        if (isinstance(node, LeafNode)):
            return node.prediction.index[0]
        else:
            # First check if the value type is numeric, then we do inequality check for numbers
            # If the value is not numeric then simply compare using ==
            value = row[node.feature]
            if ((isinstance(value, int) or isinstance(value, float)) and (value >= node.featureValue)) or \
                (value == node.featureValue):
                return self._classifyOneSample(row, node.left)
            else:
                return self._classifyOneSample(row, node.right)
    
    def _buildTree(self, dataFrame):
        ''' Build the trained decision tree with the given data frame '''
        feature, featureValue, infoGain = self.findBestFeature(dataFrame)
        print("Best feature:", feature, "Best gain:", infoGain)
        if (infoGain == 0):
            return LeafNode(dataFrame[self._targetFeature].value_counts())

        trueData, falseData = self.partition(dataFrame, feature, featureValue)

        left = self._buildTree(trueData)
        right = self._buildTree(falseData)

        return DecisionNode(left, right, feature, featureValue)
    
