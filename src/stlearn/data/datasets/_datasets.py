import os
import numpy as np
import lightkurve as lk

from astropy.units import cds
from multiprocessing import Pool
from typing import Dict, List
from stlearn.data.datasets.base import StellarDataset
from tsfresh.utilities.dataframe_functions import impute
from stlearn.conventions import KeplerQ9 as keplerq9_classes
from stlearn.data.preprocessing import pad_sequences


class KeplerBase(StellarDataset):
    """
    Kepler Q9 base dataset. To create a Kepler dataset one must extend this
    class.
    """
    TYPES = [
        key for key in keplerq9_classes.__dict__ if not key.startswith('__')
    ]

    def __init__(self):
        super().__init__()

        self.__url = None
        self.__temp = 'temp'
        self.__name = None
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

    def as_lightkurve(
        self, collection: Dict[str, np.ndarray]
    ) -> Dict[str, List]:
        if self.__id_dict is None:
            msg = "'get_ids' needs to be called first."
            raise ValueError(msg)

        lk_dict = {}
        for key in collection:
            lk_dict[key] = []
            for i, seq in enumerate(collection[key]):
                
                id = self.__id_dict[key][i].replace('.txt', '')

                lightcurve = lk.TessLightCurve(
                    time=seq[:, 0],
                    flux=seq[:, 1],
                    flux_err=seq[:, 2],
                    flux_unit=cds.ppm,
                    time_format='jd',
                    time_scale='tdb',
                    targetid=id
                )

                lk_dict[key].append(lightcurve)

        return lk_dict

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


class KeplerQ9(KeplerBase):
    def __init__(self):
        super().__init__()

        self.__url = 'https://tasoc.dk/pipeline/starclass_trainingsets/keplerq9.zip'
        self.__name = 'keplerq9'


class KeplerQ9V2(KeplerBase):
    def __init__(self):
        super().__init__()

        self.__url = 'https://tasoc.dk/pipeline/starclass_trainingsets/keplerq9v2.zip'
        self.__name = 'keplerq9v2'


class KeplerQ9V3(KeplerBase):
    def __init__(self):
        super().__init__()

        self.__url = 'https://tasoc.dk/pipeline/starclass_trainingsets/keplerq9v3.zip'
        self.__name = 'keplerq9v3'