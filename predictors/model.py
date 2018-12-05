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
    def __init__(self, targetFeature, probThresholds=None):
        self._targetFeature = targetFeature
        self._probThresholds = probThresholds

    @property
    def targetFeature(self):
        return self._targetFeature
    
    @targetFeature.setter
    def targetFeature(self, value):
        self._targetFeature = value

    @property
    def probThresholds(self):
        return self._probThresholds
    
    @probThresholds.setter
    def probThresholds(self, value):
        self._probThresholds = value

    def train(self, dataFrame, **kwargs):
        ''' '''
        raise NotImplementedError("Method \"train\" not implemented")

    def classify(self, dataFrame, **kwargs):
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
