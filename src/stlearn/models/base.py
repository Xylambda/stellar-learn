"""
Base class to build models under stellar-learn framework.
"""
from stlearn import io


class ModelBase:
    """Base model class.

    Any model that is to be used in stellar-learn needs to extend this class
    and fill the necessary methods.

    """

    def __init__(self) -> None:

        self.__fitted__ = False
        self.model = None

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
        io.save_pickle(obj=self.model, path=path)

    def load_model(self, path):
        self.model = io.load_pickle(path=path)

    def _check_fitted(self):
        if not self.__fitted__:
            return ValueError("Model has not been optimized.")
