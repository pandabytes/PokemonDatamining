import numpy as np
import pandas as pd
import decorators as decor
from multiprocessing.pool import ThreadPool
from model import SupervisedModel, FeatureType

class TreeNode:
    ''' Base class for representing tree node '''
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
    def __init__(self, prediction, probability):
        super().__init__(None, None)
        self.prediction = prediction
        self.probability = probability

class DecisionTree(SupervisedModel):
    ''' Decision Tree classifiier. It takes a target feature as the predicted feature.
        It contains a reference to a TreeNode object after it is trained.
        Use Gini Impurity and build a binary tree.

        This class uses the example from here as a base https://github.com/random-forests/tutorials/blob/master/decision_tree.ipynb
    '''
    def __init__(self, targetFeature, maxDepth=3):
        super().__init__(targetFeature)
        self._trainedRootNode = None
        self._maxDepth = maxDepth
        
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
        leftData, rightData = None, None
        featureType = super()._getFeatureType(dataFrame, feature)

        if (featureType == FeatureType.Continuous):
            assert type(value) == int or type(value) == float, "Numeric feature must be passed with a numeric value"
            leftData, rightData = self.partitionContinuous(dataFrame, feature, value)

        elif (featureType == FeatureType.Categorical):
            assert type(value) == str, "Categorical feature must be passed with a string value"
            leftData, rightData = self.partitionDiscreteBinary(dataFrame, feature, value)
            
        return leftData, rightData
    
    def partitionContinuous(self, dataFrame, feature, quantileValue):
        ''' Partition continous values with a given feature and quantile value. '''
        leftData = dataFrame[dataFrame[feature] >= quantileValue]
        rightData = dataFrame[dataFrame[feature] < quantileValue]  
        return leftData, rightData
    
    def partitionDiscrete(self, dataFrame, feature):
        ''' Partition a categorical feature into x number of categorical value of the given feature '''
        partitions = []
        for value in dataFrame[feature].unique():
            partitions.append(dataFrame[dataFrame[feature] == value])
        return partitions
                
    def partitionDiscreteBinary(self, dataFrame, feature, value):
        ''' Partition a categorical feature into 2 sub-panda frames '''
        leftData = dataFrame[dataFrame[feature] == value]
        rightData = dataFrame[dataFrame[feature] != value]  
        return leftData, rightData
    
    def findBestFeature(self, dataFrame, quantiles=[0.2, 0.4, 0.6, 0.8]):
        ''' Find the best feature to split the given data frame. Quantiles are optional and 
            are only used for continous features
        '''
        bestGain = 0.0
        currentImpurity = self.giniImpurity(dataFrame)
        bestFeature = None
        bestFeatureValue = None
        features = dataFrame.loc[:, dataFrame.columns != self._targetFeature].columns.values

        for feature in features:
            featureType = super()._getFeatureType(dataFrame, feature)
            
            if (featureType == FeatureType.Continuous):
                values = pd.Series([i for i in dataFrame[feature]])
                quantileValues = values.quantile(quantiles, "linear")

                # Find the best quantile value
                for quantileValue in quantileValues:
                    leftData, rightData = self.partition(dataFrame, feature, quantileValue)

                    # If one of the splits has no elements, then the split is trivial
                    if (len(leftData) == 0 or len(rightData) == 0):
                        continue
                    
                    infoGain = self.informationGain(leftData, rightData, currentImpurity)
                    if (infoGain > bestGain):
                        bestGain = infoGain
                        bestFeature = feature
                        bestFeatureValue = quantileValue

            elif (featureType == FeatureType.Categorical):
                for featureValue in dataFrame[feature].unique():
                    leftData, rightData = self.partition(dataFrame, feature, featureValue)

                    if (len(leftData) == 0 or len(rightData) == 0):
                        continue
                        
                    infoGain = self.informationGain(leftData, rightData, currentImpurity)
                    if (infoGain > bestGain):
                        bestGain = infoGain
                        bestFeature = feature
                        bestFeatureValue = featureValue
                    
        return bestFeature, bestFeatureValue, bestGain
        
    def countLeafNodes(self):
        ''' '''
        return self._countLeafNodes(self._trainedRootNode)

    def countTreeDepth(self):
        ''' '''
        return self._countTreeDepth(self._trainedRootNode)

    def _countLeafNodes(self, node):
        ''' '''
        if (isinstance(node, LeafNode)):
            print(node.prediction, "\n")
            return 1
        else:
            return self._countLeafNodes(node.left) + self._countLeafNodes(node.right)

    def _countTreeDepth(self, node):
        if (node.left == None and node.right == None):
            return 0
        else:
            return 1 + max(self._countTreeDepth(node.left), self._countTreeDepth(node.right))

    @decor.elapsedTime
    def train(self, dataFrame):
        ''' Train the decision tree with the given data frame input '''
        self._trainedRootNode = self._buildTree(dataFrame, 0)

    def classify(self, dataFrame):
        ''' Classify the given data frame '''
        assert self._targetFeature not in dataFrame.columns.values, "Test data must not contain the target feature \"%s\"" % self._targetFeature
        result = {}
        for i, row in dataFrame.iterrows():
            prediction = self._classifyOneSample(row, self._trainedRootNode)
            result[i] = prediction
            pass
        return pd.Series(result)

    def _classifyOneSample(self, row, node):
        ''' Classfiy one sample '''
        if (isinstance(node, LeafNode)):
            return node.prediction, node.probability
        else:
            # First check if the value type is numeric, then we do inequality check for numbers
            # If the value is not numeric then simply compare using ==
            value = row[node.feature]
            if ((isinstance(value, int) or isinstance(value, float)) and (value >= node.featureValue)) or \
                (value == node.featureValue):
                return self._classifyOneSample(row, node.left)
            else:
                return self._classifyOneSample(row, node.right)
    
    def _buildTreeThread(self, dataFrame):
        ''' Build the trained decision tree using multithreading. This creates 2 working thread.
            Each one is responsible for the left and right branch of the tree.

            @TODO: UNUSED AND IMPCOMPLETE
        '''
        feature, featureValue, infoGain = self.findBestFeature(dataFrame)    
        if (infoGain == 0):
            return LeafNode(dataFrame[self._targetFeature].value_counts())

        leftData, rightData = self.partition(dataFrame, feature, featureValue)

        # Start the threads asynchronously
        pool = ThreadPool(processes=2)
        t1 = pool.apply_async(self._buildTree, (leftData, 0))
        t2 = pool.apply_async(self._buildTree, (rightData, 0))

        # Waiting for threads to complete
        t1.wait()
        t2.wait()
        return DecisionNode(t1.get(), t2.get(), feature, featureValue)

    def _buildTree(self, dataFrame, depth):
        ''' Build the trained decision tree with the given data frame '''
        predictionCount = dataFrame[self._targetFeature].value_counts()
        bestLabel, count = max(predictionCount.items(), key=lambda x: x[1])

        # Stop splitting once the max depth of the tree is reached
        if (depth >= self._maxDepth):
            return LeafNode(prediction=bestLabel, probability=count / predictionCount.sum())

        feature, featureValue, infoGain = self.findBestFeature(dataFrame)
        if (infoGain == 0):
            return LeafNode(prediction=bestLabel, probability=count / predictionCount.sum())

        leftData, rightData = self.partition(dataFrame, feature, featureValue)

        left = self._buildTree(leftData, depth + 1)
        right = self._buildTree(rightData, depth + 1)

        return DecisionNode(left, right, feature, featureValue)
    
