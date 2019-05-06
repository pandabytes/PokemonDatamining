import pandas as pd
from models.model import Model

# Define this supervised base class before 
# import all the derived supervised classes
class SupervisedModel(Model):
    ''' Model the abstract base class structure but this class can still
        be instanitiated. Abstract methods throw an exception if 
        they are not overriden.
    '''
    def __init__(self, targetFeature):
        ''' Constructor '''
        self._targetFeature = targetFeature

    @property
    def targetFeature(self) -> str:
        return self._targetFeature
    
    @targetFeature.setter
    def targetFeature(self, value: str):
        self._targetFeature = value

    def classify(self, dataFrame: pd.DataFrame, **kwargs):
        ''' '''
        if (self._targetFeature in dataFrame.columns.values):
            raise ValueError("Test data must not contain the target feature \"%s\"" % self._targetFeature)

    def _createResultDataFrame(self, predictions: [str], probabilities: [float], indexes=None):
        ''' '''
        return pd.DataFrame({"Prediction": predictions, "Probability": probabilities}, index=indexes)


# Import all the supervised models here
from .dt.decisionTree       import DecisionTree
from .nb.naiveBayes         import NaiveBayes
from .knn.kNearestNeighbors import KNearestNeighbors