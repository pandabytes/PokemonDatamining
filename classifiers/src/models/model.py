import numpy as np
import pandas as pd

class FeatureType:
    Categorical = 0
    Continuous = 1
    Boolean = 2

class Model:
    ''' '''

    @property
    def name(self) -> str:
        return "Unknown Model"

    def train(self, dataFrame: pd.DataFrame, **kwargs):
        ''' Train the model with given training data '''
        raise NotImplementedError("Method \"train\" is not implemented for model \"%s\"" % self.name)

    def clear(self):
        ''' Clear the current state and all data of the model.
            This doesn't clear the properties of the model, however.
        '''
        raise NotImplementedError("Method \"clear\" is not implemented for model \"%s\"" % self.name)

    def _getFeatureType(self, dataFrame: pd.DataFrame, feature: str) -> FeatureType:
        ''' '''
        featureType = dataFrame.dtypes[feature].type
        if (featureType == np.int32 or featureType == np.float32 or \
            featureType == np.int64 or featureType == np.float64):
            return FeatureType.Continuous
        elif (featureType == np.object_ or (featureType == np.bool_)):
            return FeatureType.Categorical
        else:
            raise ValueError("Invalid feature %s type" % featureType)

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

class UnsupervisedModel(Model):
    ''' '''
    def __init__(self):
        ''' Constructor '''
        pass


