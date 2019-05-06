import numpy as np
import pandas as pd
import utils.decorators as decor

from graphviz import Digraph
from multiprocessing.pool import ThreadPool

from .treeNode import TreeNode, DecisionNode, LeafNode
from models.model import FeatureType
from models.supervised import SupervisedModel

class DecisionTree(SupervisedModel):
    ''' Decision Tree classifiier. It takes a target feature as the predicted feature.
        It contains a reference to a TreeNode object after it is trained.
        Use Gini Impurity and build a binary tree.

        This class uses the example from here as a base https://github.com/random-forests/tutorials/blob/master/decision_tree.ipynb
    '''

    ContinousSplitMethods = ["k-tile", "mean"]

    def __init__(self, targetFeature: str, continuousSplitmethod: str="k-tile", maxDepth: int=3, filePath: str="tree"):
        ''' Constructor '''
        if (continuousSplitmethod not in DecisionTree.ContinousSplitMethods):
            raise ValueError("Continuous split method \"" + continuousSplitmethod + "\" is not supported")
        elif (maxDepth < 0):
            raise ValueError("Max depth must be at least 0")

        super().__init__(targetFeature)
        self._trainedRootNode = None
        self._maxDepth = maxDepth
        self._continuousSplitmethod = continuousSplitmethod
        self._filePath = filePath
        self._nodeId = 0 # Use to keep track of the nodes in DiGraph
        self._diGraph = Digraph("G", filename=filePath, format="png")

    @property
    def name(self) -> str:
        return "Decision Tree"

    @property
    def maxDepth(self) -> int:
        ''' '''
        return self._maxDepth

    @maxDepth.setter
    def maxDepth(self, value: int):
        ''' '''
        if (value < 0):
            raise ValueError("Max depth must be at least 0")
        self._maxDepth = value

    @property
    def continuousSplitMethod(self) -> str:
        ''' '''
        return self._continuousSplitmethod

    @continuousSplitMethod.setter
    def continuousSplitMethod(self, value: str):
        ''' '''
        if (value not in DecisionTree.ContinousSplitMethods):
            raise ValueError("Continuous split method \"" + value + "\" is not supported")
        self._continuousSplitmethod = value

    @property
    def numLeafNodes(self) -> int:
        ''' Get the number of leaf nodes in the tree '''
        return self._countLeafNodes(self._trainedRootNode)
   
    @property
    def depth(self) -> int:
        ''' Get the depth of the tree '''
        return self._countTreeDepth(self._trainedRootNode)

    def clear(self):
        ''' Clear the current state and all data of the model.
            This doesn't clear the properties of the model, however.
        '''
        self._trainedRootNode = None
        self._diGraph.clear()

    def informationGain(self, left: pd.DataFrame, right: pd.DataFrame, currentImpurity: float) -> float:
        ''' Compute the information gain of the split 

            @left: the left partition of the data
            @right: the right partition of the data
            @currentImpurity: impurity value of the left and right partition data combined
            @return: the information gain obtained from resulting left & right partition
        '''
        p = len(left) / float(len(left) + len(right))
        childrenImpurity =  (p * self.giniImpurity(left)) + ((1-p) * self.giniImpurity(right))
        return currentImpurity - childrenImpurity
    
    def giniImpurity(self, dataFrame: pd.DataFrame) -> float:
        ''' Compute the Gini Impurity of the given data frame 
        
            @dataFrame: data frame object
            @return: gini impurity value of the given data frame
        '''
        labelCounts = dataFrame[self._targetFeature].value_counts()
        impurity = 1
        for label in labelCounts.index:
            probability = labelCounts[label] / float(len(dataFrame))
            impurity -= probability**2
        return impurity
    
    def partition(self, dataFrame: pd.DataFrame, feature: str, value: object) -> (pd.DataFrame, pd.DataFrame):
        ''' Partition the given data frame into 2 sub-data frames by the given feature and its value 
        
            @dataFrame: the data frame object
            @feature: the featured that is used as the splitting feature
            @value: value of the given feature
            @return: a tuple of 2 partitions after the split
        '''
        leftData, rightData = None, None
        featureType = super()._getFeatureType(dataFrame, feature)

        if (featureType == FeatureType.Continuous):
            if not (DecisionTree.isContinuous(value)):
                raise ValueError("Numeric feature must be passed with a numeric value")
            leftData, rightData = self.partitionContinuous(dataFrame, feature, value)

        elif (featureType == FeatureType.Categorical):
            if not (DecisionTree.isCategorical(value)):
                raise ValueError("Categorical feature must be passed with a string or boolean value")
            leftData, rightData = self.partitionDiscreteBinary(dataFrame, feature, value)
            
        return leftData, rightData
    
    def partitionContinuous(self, dataFrame: pd.DataFrame, feature: str, value: float) -> (pd.DataFrame, pd.DataFrame):
        ''' Partition continous values with a given feature and quantile value. 
        
            @dataFrame: the data frame object
            @feature: the featured that is used as the splitting feature
            @value: value of the given feature
            @return: a tuple of 2 partitions after the split
        '''
        leftData = dataFrame[dataFrame[feature] < value]
        rightData = dataFrame[dataFrame[feature] >= value]  
        return leftData, rightData
    
    def partitionDiscrete(self, dataFrame: pd.DataFrame, feature: str) -> (pd.DataFrame, pd.DataFrame):
        ''' Partition a categorical feature into x number of categorical value of the given feature 
        
            @dataFrame: the data frame object
            @feature: the featured that is used as the splitting feature
            @return: a list of x partitions
        '''
        partitions = []
        for value in dataFrame[feature].unique():
            partitions.append(dataFrame[dataFrame[feature] == value])
        return partitions
                
    def partitionDiscreteBinary(self, dataFrame: pd.DataFrame, feature: str, value: str) -> (pd.DataFrame, pd.DataFrame):
        ''' Partition a categorical feature into 2 sub-panda frames
        
            @dataFrame: the data frame object
            @feature: the featured that is used as the splitting feature
            @value: value of the given feature
            @return: a tuple of 2 partitions after the split
        '''
        leftData = dataFrame[dataFrame[feature] == value]
        rightData = dataFrame[dataFrame[feature] != value]  
        return leftData, rightData
    
    def findBestFeature(self, dataFrame: pd.DataFrame, quantiles: [int]=[0.2, 0.4, 0.6, 0.8]) -> (str, object, float):
        ''' Find the best feature to split the given data frame. Quantiles are optional and 
            are only used for continous features.

            @dataFrame: the data frame object
            @quantiles (optional): list of quantiles to test against to find the best quantile. 
            @return: a tuple of the best feature to be split, its corresponding value, and the best information gain
        '''
        bestGain = 0.0
        currentImpurity = self.giniImpurity(dataFrame)
        bestFeature = None
        bestFeatureValue = None
        features = dataFrame.loc[:, dataFrame.columns != self._targetFeature].columns.values

        for feature in features:
            featureType = super()._getFeatureType(dataFrame, feature)
            
            if (featureType == FeatureType.Continuous):
                infoGain, featureValue = 0.0, 0.0

                if (self._continuousSplitmethod == "k-tile"):
                    infoGain, featureValue = self._splitByKTile(dataFrame, feature, quantiles)
                elif (self._continuousSplitmethod == "mean"):
                    infoGain, featureValue = self._splitByMean(dataFrame, feature)

                # Store the current best values
                if (infoGain > bestGain):
                    bestGain = infoGain
                    bestFeature = feature
                    bestFeatureValue = featureValue

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

    @decor.elapsedTime
    def train(self, dataFrame: pd.DataFrame, **kwargs):
        ''' Train the decision tree with the given data frame input. Build the tree.
        
            @dataFrame: the data frame object
        '''
        self.clear()
        self._trainedRootNode = self._buildTree(dataFrame, 0)

    def classify(self, dataFrame: pd.DataFrame, **kwargs):
        ''' Classify the input data frame and return a data frame with 2 columns: Prediction and Probability.
            Prediction column denotes the predicted label of a data point and Probability column denotes the
            probability that the prediction is drawn from.

            @dataFrame: the data frame object
        '''
        super().classify(dataFrame, **kwargs)
        predictions = []
        probabilities = []
        for i, row in dataFrame.iterrows():
            prediction, probability = self._classifyOneSample(row, self._trainedRootNode)
            predictions.append(prediction)
            probabilities.append(probability)
            
        return self._createResultDataFrame(predictions, probabilities, dataFrame.index)

    def getTreeGraph(self, regenerate: bool) -> Digraph:
        ''' Get the graph object representing this decision tree
        
            @regenerate: True if we want to regenerate the graph object. False otherwise
            @return: a Digraph object from graphviz library
        '''
        if (regenerate):
            self._diGraph.clear()
            self._nodeId = 0
            self._generateGraph(self._trainedRootNode)
        return self._diGraph

    def _createEdgeLabel(self, branch: str, featureValue: object) -> str:
        ''' Create edge label according to the type of the feature and its value 
        
            @branch: value must be "left" or "right". Case-sensitive
            @featureValue: feature value to be displayed in the edge label
            @return: the edge label 
        '''
        if (branch != "left") and (branch != "right"):
            raise ValueError("Argument branch must be either \"left\" or \"right\"")

        if (DecisionTree.isCategorical(featureValue)):
            if (branch == "left"):
                return "yes"
            else:
                return "no"

        if (DecisionTree.isContinuous(featureValue)):
            if (branch == "left"):
                return "< {0:.2f}".format(featureValue)
            else:
                return ">= {0:.2f}".format(featureValue)

        raise ValueError("Feature type \"{0}\" is not str, int, bool, or float".format(featureValueType))

    def _generateGraph(self, node: TreeNode):
        ''' Generate the decision tree graph. Assign unique id to each node 
            starting from the root, left side, and then right side .
        
            @node: the root node of the decision tree
        '''
        if (node is None):
            return

        left = node.left
        right = node.right
        nodeId = self._nodeId

        decisionNodeLabelFormat = "{0}\nValue: {1}"
        leafNodeLabelFormat = "Prediction: {0}\nProbability: {1:.2f}"
        decisionNodeLabelFunc = lambda feature, value : decisionNodeLabelFormat.format(feature,
                                                                                       value if (type(value) is str or \
                                                                                                 type(value) is bool or \
                                                                                                 type(value) is np.bool_) 
                                                                                             else round(value, 2))

        # If the root node is a leaf node
        if (type(node) is LeafNode):
            nodeLabel = leafNodeLabelFormat.format(node.prediction, node.probability)
            self._addNode(LeafNode, nodeId, nodeLabel)
            return
        else:
            nodeLabel = decisionNodeLabelFunc(node.feature, node.featureValue)
            self._addNode(DecisionNode, nodeId, nodeLabel)

        if (type(left) is LeafNode and type(right) is LeafNode):
            leftLabel = leafNodeLabelFormat.format(left.prediction, left.probability)
            rightLabel = leafNodeLabelFormat.format(right.prediction, right.probability)

            # Get left and right node id
            leftId = self._nodeId + 1
            rightId = self._nodeId + 2
            self._nodeId += 2

            self._addNode(LeafNode, leftId, nodeLabel)
            self._addNode(LeafNode, rightId, nodeLabel)

            self._addEdge(nodeId, leftId, self._createEdgeLabel("left", node.featureValue))
            self._addEdge(nodeId, rightId, self._createEdgeLabel("right", node.featureValue))
            
        elif (type(left) is LeafNode):
            leftLabel = leafNodeLabelFormat.format(left.prediction, left.probability)
            rightLabel = decisionNodeLabelFunc(right.feature, right.featureValue)

            # Assign id to the left node first
            leftId = self._nodeId + 1
            self._addNode(LeafNode, leftId, leftLabel)
            self._nodeId += 1

            # Then assign id to the right node recursively
            rightId = self._nodeId + 1
            self._addNode(DecisionNode, rightId, rightLabel)
            self._nodeId += 1
            self._generateGraph(right)

            self._addEdge(nodeId, leftId, self._createEdgeLabel("left", node.featureValue))
            self._addEdge(nodeId, rightId, self._createEdgeLabel("right", node.featureValue))

        elif (type(right) is LeafNode):
            leftLabel = decisionNodeLabelFunc(left.feature, left.featureValue)
            rightLabel = leafNodeLabelFormat.format(right.prediction, right.probability)

            # Assign id to the left node first recursively
            leftId = self._nodeId + 1
            self._addNode(DecisionNode, leftId, leftLabel)
            self._nodeId += 1
            self._generateGraph(left)

            # Then assig id to the right node
            # Don't need to add 1 after each _generateGraph call. It's handled at the end of the method
            rightId = self._nodeId 
            self._addNode(LeafNode, rightId, rightLabel)
            
            self._addEdge(nodeId, leftId, self._createEdgeLabel("left", node.featureValue))
            self._addEdge(nodeId, rightId, self._createEdgeLabel("right", node.featureValue))

        else:
            leftLabel = decisionNodeLabelFunc(left.feature, left.featureValue)
            rightLabel = decisionNodeLabelFunc(right.feature, right.featureValue)

            # Assign id to the left node first recursively
            leftId = self._nodeId + 1
            self._addNode(DecisionNode, leftId, leftLabel)
            self._nodeId += 1
            self._generateGraph(left)

            # Then assign id to the right node recursively
            # Don't need to add 1 after each _generateGraph call. It's handled at the end of the method
            rightId = self._nodeId
            self._addNode(DecisionNode, rightId, rightLabel)
            self._generateGraph(right)

            self._addEdge(nodeId, leftId, self._createEdgeLabel("left", node.featureValue))
            self._addEdge(nodeId, rightId, self._createEdgeLabel("right", node.featureValue))

        self._nodeId += 1

    def _addEdge(self, fromId: int, toId: int, nodeLabel: str):
        ''' '''
        self._diGraph.edge(str(fromId), str(toId), label=nodeLabel)

    def _addNode(self, nodeType: TreeNode, nodeId: int, nodeLabel: str):
        ''' '''
        if (nodeType is LeafNode):
            self._diGraph.node(str(nodeId), nodeLabel, color="red")
        elif (nodeType is DecisionNode):
            self._diGraph.node(str(nodeId), nodeLabel)
        else:
            raise ValueError("Invalid node type \"{0}\"".format(nodeType, LeafNode, DecisionNode))

    def _classifyOneSample(self, row: pd.Series, node: TreeNode) -> (str, float):
        ''' Classfiy one sample 

            @row: row of a data frame
            @node: the root node of the decision tree
            @return: the prediction and probability of that prediction
        '''
        if (type(node) is LeafNode):
            return node.prediction, node.probability
        else:
            # First check if the value type is numeric, then we do inequality check for numbers
            # If the value is not numeric then simply compare using ==
            value = row[node.feature]
            if (DecisionTree.isContinuous(value) and value < node.featureValue) or (value == node.featureValue):
                return self._classifyOneSample(row, node.left)
            else:
                return self._classifyOneSample(row, node.right)
    
    def _buildTreeThread(self, dataFrame, depth):
        ''' Build the trained decision tree using multithreading. This creates 2 working thread.
            Each one is responsible for the left and right branch of the tree.

            @TODO: UNUSED AND IMPCOMPLETE
        '''
        predictionCount = dataFrame[self._targetFeature].value_counts()

        # Stop splitting once the max depth of the tree is reached
        if (depth >= self._maxDepth):
            bestLabel, bestLabelCount = max(predictionCount.items(), key=lambda x: x[1])
            bestProb = float(bestLabelCount) / sum(predictionCount)
            return LeafNode(prediction=bestLabel, probability=bestProb)

        # Stop splitting if there's no more information to gain
        feature, featureValue, infoGain = self.findBestFeature(dataFrame)
        if (infoGain == 0):
            bestLabel, bestLabelCount = max(predictionCount.items(), key=lambda x: x[1])
            bestProb = float(bestLabelCount) / sum(predictionCount)
            return LeafNode(prediction=bestLabel, probability=bestProb)

        leftSubTree = None
        rightSubTree = None
        leftData, rightData = self.partition(dataFrame, feature, featureValue)
        if (depth == 0):
            # Start the threads asynchronously
            pool = ThreadPool(processes=2)
            t1 = pool.apply_async(self._buildTreeThread, (leftData, depth + 1))
            t2 = pool.apply_async(self._buildTreeThread, (rightData, depth + 1))
            print("waiting for threads")
            t1.wait()
            t2.wait()
            leftSubTree = t1.get()
            rightSubTree = t2.get()
        else:
            leftSubTree = self._buildTreeThread(leftData, depth + 1)
            rightSubTree = self._buildTreeThread(rightData, depth + 1)

        return DecisionNode(leftSubTree, rightSubTree, feature, featureValue)

    def _buildTree(self, dataFrame: pd.DataFrame, depth: int) -> TreeNode:
        ''' Build the trained decision tree with the given data frame
        
            @dataFrame: data frame object
            @depth: that maximum depth of the tree. Stop building the tree once
                    the depth of the tree reaches to this value.
            @return: the root node of the decision tree
        '''
        predictionCount = dataFrame[self._targetFeature].value_counts()

        # Stop splitting once the max depth of the tree is reached
        if (depth >= self._maxDepth):
            bestLabel, bestLabelCount = max(predictionCount.items(), key=lambda x: x[1])
            bestProb = float(bestLabelCount) / sum(predictionCount)
            return LeafNode(prediction=bestLabel, probability=bestProb)

        # Stop splitting if there's no more information to gain
        feature, featureValue, infoGain = self.findBestFeature(dataFrame)
        if (infoGain == 0):
            bestLabel, bestLabelCount = max(predictionCount.items(), key=lambda x: x[1])
            bestProb = float(bestLabelCount) / sum(predictionCount)
            return LeafNode(prediction=bestLabel, probability=bestProb)

        leftData, rightData = self.partition(dataFrame, feature, featureValue)

        left = self._buildTree(leftData, depth + 1)
        right = self._buildTree(rightData, depth + 1)

        return DecisionNode(left, right, feature, featureValue)

    def _countLeafNodes(self, node: TreeNode) -> int:
        ''' Helper function for counting leaf nodes 
        
            @node: root node of the decision tree node
            @return: number of leaf nodes
        '''
        if (node is None):
            return 0
        elif (type(node) is LeafNode):
            return 1
        else:
            return self._countLeafNodes(node.left) + self._countLeafNodes(node.right)

    def _countTreeDepth(self, node: TreeNode) -> int:
        ''' Helper function for counting the tree depth 
        
            @node: root node of the decision tree
            @return: the depth of the decision tree
        '''
        if (node is None) or (node.left == None and node.right == None):
            return 0
        else:
            return 1 + max(self._countTreeDepth(node.left), self._countTreeDepth(node.right))

    def _splitByKTile(self, dataFrame: pd.DataFrame, feature: str, quantiles: [int]) -> (float, float):
        ''' Split continuous feature by using k-tile method. 
        
            @dataFrame: data frame object
            @feature: the feature that is used as the splitting point
            @quantiles: list of quantiles use for determining the best quantile in that list
            @return: tuple containing best information gain and the best quantile value
        '''
        bestGain = 0.0
        bestQuantileValue = None
        currentImpurity = self.giniImpurity(dataFrame)
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
                bestQuantileValue = quantileValue
        
        return bestGain, bestQuantileValue

    def _splitByMean(self, dataFrame: pd.DataFrame, feature: str) -> (float, float):
        ''' Split continuous feature by using mean method.
        
            @dataFrame: data frame object
            @feature: the feature that is used as the splitting point
            @return: tuple containing best information gain and the mean
        '''
        # Use the the mean as the splitting point
        mean = dataFrame[feature].mean()
        currentImpurity = self.giniImpurity(dataFrame)

        leftData, rightData = self.partition(dataFrame, feature, mean)
        infoGain = self.informationGain(leftData, rightData, currentImpurity)
        return infoGain, mean

