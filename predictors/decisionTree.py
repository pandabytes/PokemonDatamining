# import gc
# import math
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA

class DecisionTree:
    def __init__(self, data, targetColumName):
        self._data = data
        self._targetColumnName = targetColumName
    
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, value):
        self._data = value
        
    @property
    def targetColumName(self):
        return self._data
    
    @targetColumName.setter
    def data(self, value):
        self._targetColumName = value
        
    def informationGain(self, left, right, currentImpurity):
        ''''''
        p = len(left) / float(len(left) + len(right))
        childrenImpurity =  (p * self.giniImpurity(left)) + ((1-p) * self.giniImpurity(right))
        return currentImpurity - childrenImpurity
    
    def giniImpurity(self, data):
        ''' '''
        labelCounts = data[self._targetColumnName].value_counts()
        impurity = 1
        
        for label in labelCounts.index:
            probability = labelCounts[label] / float(len(data))
            impurity -= probability**2
        return impurity
    
#     def partition(self, columnName):
#         partitions = {}
        
#         if (isinstance(self._data.dtypes[columnName], object)):
#             pass
#         elif (isinstance(self._data.dtypes[columnName], int) or isinstance(self._data.dtypes[columnName], float)):
#             pass
#         else:
#             raise Exception("Invalid column type")
            
#         return partitions
    
    def partitionContinuous(self, columnName, quantile):
        sortedValues = pd.Series(sorted([i for i in self._data[columnName].unique()]))
        quantileValue = sortedValues.quantile(quantile, "linear")
        print("Quantile:", quantile, "--->", quantileValue)
        trueData = self._data[self._data[columnName] >= quantileValue]
        falseData = self._data[self._data[columnName] < quantileValue]  
        
        return trueData, falseData
    
    def partitionDiscrete(self, columnName):
        partitions = []
        for value in self._data[columnName].unique():
            partitions.append(self._data[self._data[columnName] == value])
        return partitions
                
    def partitionDiscreteBinary(self, columnName, value):
        trueData = self._data[self._data[columnName] == value]
        falseData = self._data[self._data[columnName] != value]  

        return trueData, falseData      
