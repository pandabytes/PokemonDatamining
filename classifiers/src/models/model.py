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

    @staticmethod
    def isContinuous(value: object) -> bool:
        ''' Check if the type of the given value is continous aka is its type an int or float.
        
            @value: value to be type-checked
            @return: true if the type of the given value is int or float. False otherwise.
        '''
        valueType = type(value)
        return (valueType is int or valueType is np.int32 or valueType is np.int64 or \
                valueType is float or valueType is np.float32 or valueType is np.float64)

    @staticmethod
    def isCategorical(value: object) -> bool:
        ''' Check if the type of the given value is categorical aka is its type an string or boolean.
        
            @value: value to be type-checked
            @return: true if the type of the given value is str or bool. False otherwise.
        '''
        valueType = type(value)
        return (valueType is str or valueType is bool or valueType is np.bool_)
