import numpy as np
import pandas as pd

from models.model import FeatureType
from models.unsupervised import UnsupervisedModel

class Point:
    ''' '''
    def __init__(self):
        ''' Constructor '''
        pass

class Cluster:
    ''' '''
    def __init__(self, points):
        ''' Constructor '''
        self._points = points
        self._centroid = self._findCentroid()

    def _findCentroid(self):
        ''' '''
        pass

class KMeans(UnsupervisedModel):
    ''' '''
    def __init__(self):
        ''' Constructor '''
        pass