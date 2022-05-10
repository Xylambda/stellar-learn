"""
Module that contains all models to classify curves.
"""
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from stlearn.models.base import ModelBase
from mlens.ensemble import SuperLearner as SPRL


class SuperLearner(ModelBase):
    """
    Parameters
    ----------
    n_folds : int, optional, default: 2
        Number of folds to use during fitting. Note: this parameter can be
        specified on a layer-specific basis in the add method.
    scorer : object, default: None
        Scoring function. If a function is provided, base estimators will be
        scored on the training set assembled for fitting the meta estimator.
        Since those predictions are out-of-sample, the scores represent valid
        test scores. The scorer should be a function that accepts an array of
        true values and an array of
    """

    def __init__(self, n_folds=2, scorer=None) -> None:
        super().__init__()

        self.model = SPRL(folds=n_folds, scorer=scorer)
        self.model.add(RandomForestClassifier())
        self.model.add(LogisticRegression())
        self.model.add_meta(MLPClassifier())

    def fit(self, X, y):
        self.model.fit(X, y)
        self.__fitted__ = True

    def predict_class(self, X):
        self._check_fitted()
        return self.model.predict_class(X)
