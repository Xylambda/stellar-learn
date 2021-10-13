"""
Base class to build models under stellar-learn framework.
"""


class ModelBase:
    """Base model class.

    Any model that is to be used in stellar-learn needs to extend this class 
    and fill the necessary methods.
    
    """
    def __init__(self) -> None:
 
        self.__fitted__ = False

    def fit(self, X, y):
        """Fit the algorithm.

        You must specify the features X and labels y to perform the 
        optimization process.

        Parameters
        ----------
        X : numpy.array
            Features.
        y : numpy.array
            Labels.

        """
        raise NotImplementedError()

    def predict_class(self, X):
        raise NotImplementedError()

    def save_model(self, path):
        pass

    def load_model(self):
        pass