import numpy as np
import pandas as pd
import utils.decorators as decor
from graphviz import Digraph
from multiprocessing.pool import ThreadPool
from ..model import SupervisedModel, FeatureType

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
    def __init__(self, targetFeature, continuousSplitmethod="k-tile", maxDepth=3, filePath="tree"):
        ''' Constructor '''
        super().__init__(targetFeature)
        self._trainedRootNode = None
        self._maxDepth = maxDepth
        self._continuousSplitmethod = continuousSplitmethod
        self._filePath = filePath
        self._nodeId = 0
        self._diGraph = Digraph("G", filename=filePath, format="png")

    @property
    def maxDepth(self):
        return self._maxDepth

    @maxDepth.setter
    def maxDepth(self, value):
        self._maxDepth = value
        
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
            if not (type(value) == int or type(value) == float or type(value) == np.int64 or type(value) == np.float64):
                raise ValueError("Numeric feature must be passed with a numeric value")
            leftData, rightData = self.partitionContinuous(dataFrame, feature, value)

        elif (featureType == FeatureType.Categorical):
            if (type(value) != str):
                raise ValueError("Categorical feature must be passed with a string value")
            leftData, rightData = self.partitionDiscreteBinary(dataFrame, feature, value)
            
        return leftData, rightData
    
    def partitionContinuous(self, dataFrame, feature, quantileValue):
        ''' Partition continous values with a given feature and quantile value. '''
        leftData = dataFrame[dataFrame[feature] < quantileValue]
        rightData = dataFrame[dataFrame[feature] >= quantileValue]  
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
                if (self._continuousSplitmethod == "k-tile"):
                    quantileValues = dataFrame[feature].quantile(quantiles, "linear")

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
                elif (self._continuousSplitmethod == "mean"):
                    # Use the the mean as the splitting point
                    mean = dataFrame[feature].mean()
                    leftData, rightData = self.partition(dataFrame, feature, mean)
                    infoGain = self.informationGain(leftData, rightData, currentImpurity)
                    if (infoGain > bestGain):
                        bestGain = infoGain
                        bestFeature = feature
                        bestFeatureValue = mean
                else: 
                    raise ValueError("Invalid continuousSplitmethod value")

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
        ''' Return the number of leaf nodes in the tree '''
        return self._countLeafNodes(self._trainedRootNode)

    def getTreeDepth(self):
        ''' Return the depth of the tree '''
        return self._countTreeDepth(self._trainedRootNode)

    @decor.elapsedTime
    def train(self, dataFrame, **kwargs):
        ''' Train the decision tree with the given data frame input '''
        self._trainedRootNode = self._buildTree(dataFrame, 0)

    def classify(self, dataFrame, **kwargs):
        ''' Classify the input data frame and return a data frame with 2 columns: Prediction and Probability.
            Prediction column denotes the predicted label of a data point and Probability column denotes the
            probability that the prediction is drawn from.
        '''
        super().classify(dataFrame, **kwargs)
        predictions = []
        probabilities = []
        for i, row in dataFrame.iterrows():
            prediction, probability = self._classifyOneSample(row, self._trainedRootNode)
            predictions.append(prediction)
            probabilities.append(probability)
            
        return pd.DataFrame({"Prediction": predictions, "Probability": probabilities}, index=dataFrame.index)

    def getTreeGraph(self):
        ''' '''
        self._diGraph.clear()
        self._nodeId = 0
        self._generateGraph(self._trainedRootNode)
        return self._diGraph

    def _createEdgeLabel(self, branch, feature, featureValue):
        ''' Create edge label according to the type of the feature and its value '''
        if (isinstance(featureValue, str)):
            if (branch == "left"):
                return "yes"
            elif (branch == "right"):
                return "no"
            raise ValueError("Argument branch must be either \"left\" or \"right\"")

        elif (isinstance(featureValue, float) or isinstance(featureValue, int)):
            return "< {0:.2f}".format(featureValue) if (branch == "left") else ">= {0:.2f}".format(featureValue)

        else:
            raise ValueError("Feature type not str, int, or float")

    def _generateGraph(self, node):
        ''' Generate the decision tree graph. Assign unique id to each node starting from the root, left side, and then right side '''
        if (node is None):
            return

        left = node.left
        right = node.right
        nodeId = self._nodeId
        nodeLabel = ""

        # If the root node is a leaf node
        if (isinstance(node, LeafNode)):
            nodeLabel = "Prediction: {0}\nProbability: {1}".format(node.prediction, node.probability)
            self._diGraph.node(str(nodeId), nodeLabel)
            return
        else:
            nodeLabel = "{0}\nValue: {1}".format(node.feature, node.featureValue)
            self._diGraph.node(str(nodeId), nodeLabel)

        if (isinstance(left, LeafNode) and isinstance(right, LeafNode)):
            leftLabel = "Prediction: {0}\nProbability: {1}".format(left.prediction, left.probability)
            rightLabel = "Prediction: {0}\nProbability: {1}".format(right.prediction, left.probability)

            # Get left and right node id
            leftId = self._nodeId + 1
            rightId = self._nodeId + 2
            self._nodeId += 2

            self._diGraph.node(str(leftId), leftLabel)
            self._diGraph.node(str(rightId), rightLabel)

            self._diGraph.edge(str(nodeId), str(leftId), label=self._createEdgeLabel("left", node.feature, node.featureValue))
            self._diGraph.edge(str(nodeId), str(rightId), label=self._createEdgeLabel("right", node.feature, node.featureValue))
            
        elif (isinstance(left, LeafNode)):
            leftLabel = "Prediction: {0}\nProbability: {1}".format(left.prediction, left.probability)
            rightLabel = "{0}\nValue: {1}".format(right.feature, right.featureValue)

            # Assign id to the left node first
            leftId = self._nodeId + 1
            self._diGraph.node(str(leftId), leftLabel)
            self._nodeId += 1

            # Then assign id to the right node recursively
            rightId = self._nodeId + 1
            self._diGraph.node(str(rightId), rightLabel)
            self._nodeId += 1
            self._generateGraph(right)

            self._diGraph.edge(str(nodeId), str(leftId), label=self._createEdgeLabel("left", node.feature, node.featureValue))
            self._diGraph.edge(str(nodeId), str(rightId), label=self._createEdgeLabel("right", node.feature, node.featureValue))

        elif (isinstance(right, LeafNode)):
            leftLabel = "{0}\nValue: {1}".format(left.feature, left.featureValue)
            rightLabel = "Prediction: {0}\nProbability: {1}".format(right.prediction, right.probability)

            # Assign id to the left node first recursively
            leftId = self._nodeId + 1
            self._diGraph.node(str(leftId), leftLabel)
            self._nodeId += 1
            self._generateGraph(left)

            # Then assig id to the right node
            # Don't need to add 1 after each _generateGraph call. It's handled at the end of the method
            rightId = self._nodeId 
            self._diGraph.node(str(rightId), rightLabel)
            
            self._diGraph.edge(str(nodeId), str(leftId), label=self._createEdgeLabel("left", node.feature, node.featureValue))
            self._diGraph.edge(str(nodeId), str(rightId), label=self._createEdgeLabel("right", node.feature, node.featureValue))
            
        else:
            leftLabel = "{0}\nValue: {1}".format(left.feature, left.featureValue)
            rightLabel = "{0}\nValue: {1}".format(right.feature, right.featureValue)

            # Assign id to the left node first recursively
            leftId = self._nodeId + 1
            self._diGraph.node(str(leftId), leftLabel)
            self._nodeId += 1
            self._generateGraph(left)

            # Then assign id to the right node recursively
            # Don't need to add 1 after each _generateGraph call. It's handled at the end of the method
            rightId = self._nodeId
            self._diGraph.node(str(rightId), rightLabel)
            self._generateGraph(right)

            self._diGraph.edge(str(nodeId), str(leftId), label=self._createEdgeLabel("left", node.feature, node.featureValue))
            self._diGraph.edge(str(nodeId), str(rightId), label=self._createEdgeLabel("right", node.feature, node.featureValue))

        self._nodeId += 1

    def _classifyOneSample(self, row, node):
        ''' Classfiy one sample '''
        if (isinstance(node, LeafNode)):
            return node.prediction, node.probability
        else:
            # First check if the value type is numeric, then we do inequality check for numbers
            # If the value is not numeric then simply compare using ==
            value = row[node.feature]
            if ((isinstance(value, int) or isinstance(value, float)) and (value < node.featureValue)) or \
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

        # Stop splitting once the max depth of the tree is reached
        if (depth >= self._maxDepth):
            bestLabel, bestProb = max(predictionCount.items(), key=lambda x: x[1])
            return LeafNode(prediction=bestLabel, probability=bestProb)

        # Stop splitting if there's no more information to gain
        feature, featureValue, infoGain = self.findBestFeature(dataFrame)
        if (infoGain == 0):
            bestLabel, bestProb = max(predictionCount.items(), key=lambda x: x[1])
            return LeafNode(prediction=bestLabel, probability=bestProb)

        leftData, rightData = self.partition(dataFrame, feature, featureValue)

        left = self._buildTree(leftData, depth + 1)
        right = self._buildTree(rightData, depth + 1)

        return DecisionNode(left, right, feature, featureValue)

    def _countLeafNodes(self, node):
        ''' Helper function for counting leaf nodes '''
        if (node is None):
            return 0
        elif (isinstance(node, LeafNode)):
            return 1
        else:
            return self._countLeafNodes(node.left) + self._countLeafNodes(node.right)

    def _countTreeDepth(self, node):
        ''' Helper function for counting the tree depth '''
        if (node is None) or (node.left == None and node.right == None):
            return 0
        else:
            return 1 + max(self._countTreeDepth(node.left), self._countTreeDepth(node.right))



