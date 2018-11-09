import gc
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class TreeNode:
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

class DecisionNode(TreeNode):
    def __init__(self, left=None, right=None, feature=None, featureValue=None):
        super().__init__(left, right)
        self.feature = feature
        self.featureValue = featureValue

class LeafNode(TreeNode):
    def __init__(self, prediction):
        super().__init__(None, None)
        self.prediction = prediction

class DecisionTree:
    def __init__(self, targetFeature):
        self._targetFeature = targetFeature
        
    @property
    def targetFeature(self):
        return self._targetFeature
    
    @targetFeature.setter
    def data(self, value):
        self._targetFeature = value
        
    def informationGain(self, left, right, currentImpurity):
        ''''''
        p = len(left) / float(len(left) + len(right))
        childrenImpurity =  (p * self.giniImpurity(left)) + ((1-p) * self.giniImpurity(right))
        return currentImpurity - childrenImpurity
    
    def giniImpurity(self, dataFrame):
        ''' '''
        labelCounts = dataFrame[self._targetFeature].value_counts()
        impurity = 1
        
        for label in labelCounts.index:
            probability = labelCounts[label] / float(len(dataFrame))
            impurity -= probability**2
        return impurity
    
    def partition(self, dataFrame, feature, value):
        trueData, falseData = None, None
        
        if (dataFrame.dtypes[feature].type == np.int64 or dataFrame.dtypes[feature].type == np.float64):
            assert type(value) == int or type(value) == float, "Numeric feature must be passed with a numeric value"
            trueData, falseData = self.partitionContinuous(dataFrame, feature, value)

        elif (dataFrame.dtypes[feature].type == np.object_):
            assert type(value) == str, "Categorical feature must be passed with a string value"
            trueData, falseData = self.partitionDiscreteBinary(dataFrame, feature, value)

        else:
            raise Exception("Invalid feature %s type" % dataFrame.dtypes[feature].type)
            
        return trueData, falseData
    
    def partitionContinuous(self, dataFrame, feature, quantileValue):
        ''' '''
        trueData = dataFrame[dataFrame[feature] >= quantileValue]
        falseData = dataFrame[dataFrame[feature] < quantileValue]  
        return trueData, falseData
    
    def partitionDiscrete(self, dataFrame, feature):
        ''' '''
        partitions = []
        for value in dataFrame[feature].unique():
            partitions.append(dataFrame[dataFrame[feature] == value])
        return partitions
                
    def partitionDiscreteBinary(self, dataFrame, feature, value):
        ''' '''
        trueData = dataFrame[dataFrame[feature] == value]
        falseData = dataFrame[dataFrame[feature] != value]  
        return trueData, falseData
    
    def findBestFeature(self, dataFrame, quantiles=[0.2, 0.4, 0.6, 0.8]):
        ''' '''
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
                        #print("Continuous -->", bestFeature, bestFeatureValue)

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
                        #print("Categorical -->", bestFeature, bestFeatureValue)
            else:
                raise Exception("Invalid feature %s type" % dataFrame.dtypes[feature].type)
                    
        return bestFeature, bestFeatureValue, bestGain
        
    def train(self, dataFrame):
        ''' '''
        feature, featureValue, infoGain = self.findBestFeature(dataFrame)
        print("Best feature", feature, "Best gain:", infoGain)
        if (infoGain == 0):
            return LeafNode(dataFrame[self._targetFeature].value_counts())

        trueData, falseData = self.partition(dataFrame, feature, featureValue)

        left = self.train(trueData)
        right = self.train(falseData)

        return DecisionNode(left, right, feature, featureValue)

    def classify(self, dataFrame, node):
        ''' '''
        result = {}
        for i, row in dataFrame.iterrows():
            result[i] = self.classifyOneSample(row, node)
        return pd.Series(result)

    def classifyOneSample(self, row, node):
        ''' '''
        if (isinstance(node, LeafNode)):
            return node.prediction.index[0]
        else:
            # First check if the value type is numeric, then we do inequality check for numbers
            # If the value is not numeric then simply compare using ==
            value = row[node.feature]
            if ((isinstance(value, int) or isinstance(value, float)) and (value >= node.featureValue)) or (value == node.featureValue):
                return self.classifyOneSample(row, node.left)
            else:
                return self.classifyOneSample(row, node.right)

    # def printTree(self, rootNode, spacing=""):
    #     ''' '''
    #     print("Beginning",type(rootNode) == LeafNode)
    #     print(id(LeafNode))
    #     print(id(type(rootNode)))
        # if (isinstance(rootNode, LeafNode)):
        #     print(spacing, "Prediction:", rootNode.prediction)
        # else:
        #     print(type(rootNode))
        #     print(spacing, "Split at feature", rootNode.feature, ". Value:", rootNode.featureValue)
        #     print(spacing, " True:")
        #     self.printTree(rootNode.left, "  ")

        #     print(spacing, " False:")
        #     self.printTree(rootNode.right, "  ")

        # for i in range(len(leafNodes)):
        #     node = leafNodes[i]
        #     print("Node", i+1)

        #     for label in node.index:
        #         print("\t", label, "--->", node[label])

    
    
    
    
