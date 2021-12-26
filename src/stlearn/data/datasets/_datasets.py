import os
import numpy as np

from multiprocessing import Pool
from stlearn.data.datasets.base import StellarDataset
from tsfresh.utilities.dataframe_functions import impute
from stlearn.conventions import KeplerQ9 as keplerq9_classes
from stlearn.data.preprocessing import pad_sequences


class KeplerQ9(StellarDataset):
    """
    Kepler Q9 dataset.
    """
    TYPES = [
        key for key in keplerq9_classes.__dict__ if not key.startswith('__')
    ]

    def __init__(self):
        super().__init__()

        self.__url = 'https://tasoc.dk/pipeline/starclass_trainingsets/keplerq9.zip'
        self.__temp = 'temp'
        self.__name = 'keplerq9'
        self.__data_collection = None

    def _read_single_class(self, path, n_processes: int) -> list:
        """Helper function.

        Read the light curves of a single stellar type.

        Parameters
        ----------
        path : path-like
            Path where to find the files for the particular class.
        n_processes : int
            Number of processes to use.

        Returns
        -------
        collection : list
        """
        # prepend folder
        _files = os.listdir(path)
        files = [path / x for x in _files]
        
        pool = Pool(n_processes)
        collection = pool.map(np.loadtxt, files)
        pool.close()
        pool.join()
        
        return collection

    def pad_collection(self, collection):
        new_collection = {}      
        for ty in self.TYPES:
            new_collection[ty] = pad_sequences(collection[ty])
            
        return new_collection

    def get_ids(self, folder):
        if self.__id_dict is None:
            id_dict = {}
            for ty in self.TYPES:
                id_dict[ty] = os.listdir(folder / ty)

            self.__id_dict = id_dict
            return self.self.__id_dict
        else:
            return self.self.__id_dict

    def from_folder(self, folder, n_processes=8):
        if self.self.__data_collection is None:            
            collection = {}
            for ty in self.TYPES:
                collection[ty] = self.read_single_class(
                    path=folder / ty, n_processes=n_processes
                )

            self.__data_collection = collection
            return self.__data_collection
        else:
            return self.__data_collection
