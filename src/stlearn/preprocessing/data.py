"""

"""


class Dataset:
    """Base class to create datasets.

    Any dataset that is to be used in 'stellar-learn' must extend this class
    and implement 'get_train_features', 'get_train_labels', 
    'get_test_features' and 'get_test_labels' methods. Additionally, validation
    methods: 'get_validation_features' and 'get_validation_features' can be
    implemented too.

    The data loading process/creation should be performed in the constructor 
    and/or in ad-hoc auxiliar methods.

    """
    def __init__(self) -> None:
        pass

    def get_train_features(self):
        """Train features getter.

        It is mandatory to fill this method.

        Returns
        -------
        array-like
            Train features set.
        """
        raise NotImplementedError(
            f"'get_train_features' method not implemented."
        )

    def get_train_labels(self):
        """Train labels getter.

        It is mandatory to fill this method.

        Returns
        -------
        array-like
            Train labels set.
        """
        raise NotImplementedError(
            f"'get_train_labels' method not implemented."
        )

    def get_test_features(self):
        """Test features getter. 

        It is mandatory to fill this method.

        Returns
        -------
        array-like
            Test features set.
        """
        raise NotImplementedError(
            f"'get_test_features' method not implemented."
        )

    def get_test_labels(self):
        """Test labels getter.

        It is mandatory to fill this method.

        Returns
        -------
        array-like
            Test labels set.
        """
        raise NotImplementedError(
            f"'get_train_labels' method not implemented."
        )

    def get_validation_features(self):
        """Validation features getter.  

        This method does not need to be filled.

        Returns
        -------
        array-like
            Train features set.
        """
        pass

    def get_validation_labels(self):
        """Validation labels getter.

        This method does not need to be filled.

        Returns
        -------
        array-like
            Validation labels set.
        """
        pass