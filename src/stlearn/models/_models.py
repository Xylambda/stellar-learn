"""
Module that contains all models to classify curves.
"""
from xgboost import XGBClassifier
from stlearn.models.base import ModelBase


class MetaClassifier(ModelBase):
    def __init__(self) -> None:
        super(MetaClassifier, self).__init__()


class SLOSH(ModelBase):
    def __init__(self) -> None:
        super(MetaClassifier, self).__init__()


class SortingHatClassifier(ModelBase):
    def __init__(self) -> None:
        super(MetaClassifier, self).__init__()


class XGBClassifier(ModelBase):
    """
    XGBoost classifier
    """
    def __init__(self) -> None:
        super(MetaClassifier, self).__init__()

        self.model = XGBClassifier(
            booster='gbtree',
            colsample_bytree=0.7,
            eval_metric='mlogloss',
            gamma=7.5,
            learning_rate=0.1,
            max_depth=6,
            min_child_weight=1,
            n_estimators=500,
            objective='multi:softmax',
            random_state=self.random_seed,
            reg_alpha=1e-5,
            subsample=0.8,
            use_label_encoder=False,
            n_jobs=1
        )

    def fit(self, X, y):
        self.model.fit(X, y)

        # mark as fitted if everything is ok
        self.__fitted__ = True

    def predict_class(self, X):
        return self.model.predict_proba(X)