import gc
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class DecisionTree:
    def __init__(self, data):
        self._data = data
    
    @property
    def data(self):
        return data
    
    @data.setter
    def data(self, value):
        self._data = value
        
    def informationGain(self):
        pass
    
    
    def giniIndex(self, targetLabel):
        ''' '''
        labelCounts = data[targetLabel].value_counts()
        impurity = 1
        
        for label in labelCounts.index:
            probability = labelCounts[label] / float(len(self._data))
            impurity -= probability**2
        return impurity
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    