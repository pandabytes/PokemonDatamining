import numpy as np

class FeatureType:
    Categorical = 0
    Continuous = 1
    Boolean = 2

class SupervisedModel:
    ''' Model the abstract base class structure but this class can still
        be instanitiated. Abstract methods throw an exception if 
        they are not overriden.
    '''
    def __init__(self, targetFeature):
        self._targetFeature = targetFeature

    @property
    def targetFeature(self):
        return self._targetFeature
    
    @targetFeature.setter
    def data(self, value):
        self._targetFeature = value

    def train(self, dataFrame, *args):
        ''' '''
        raise NotImplementedError("Method \"train\" not implemented")

    def classify(self, dataFrame, *args):
        ''' '''
        raise NotImplementedError("Method \"classify\" not implemented")

    def _getFeatureType(self, dataFrame, feature):
        ''' '''
        featureType = dataFrame.dtypes[feature].type
        if (featureType == np.int64 or featureType == np.float64):
            return FeatureType.Continuous
        elif (featureType== np.object_):
            return FeatureType.Categorical
        elif (featureType == np.object_):
            return FeatureType.Boolean
        else:
            raise ValueError("Invalid feature %s type" % featureType)
