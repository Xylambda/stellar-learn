"""
Base class for building datasets.
"""
import numpy as np
import pandas as pd
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
from urllib.request import urlopen
from typing import Dict, List, Union
from tsfresh import extract_features


class StellarDataset:
    """Base class to create datasets.

    To create a dataset one must extend this class and fill the appropiate
    methods and attributes.
    
    """
    def __init__(self):
        self.__url = None
        self.__name = None
        self.__temp = Path('temp')
        self.__data_collection = None
        self.__id_dict = None

        # Filled below
        self.tsfresh_features = None

    def download(self) -> None:
        self.__temp.mkdir(exist_ok=True)

        with urlopen(self.__url) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall(self.__temp / self.__name)

    def from_folder(
        self, folder: Union[Path, str], **kwargs
    ) -> Dict[str, List[np.ndarray]]:
        """
        Read data from the folder where the dataset is stored.
        """
        raise NotImplementedError

    def get_ids(self, folder: Union[Path, str]) -> Dict[str, List[str]]:
        """
        Retrieve the ids for each light curve.

        Returns
        -------
        id_dict : dict
            Dictionary whose keys are the star type and whose values are a list
            of all ids.
        """
        raise NotImplementedError

    def get_data_collection(self) -> dict:
        """
        Getter method that returns the sequences as a dict where each key is 
        the star type and each value is a list of all time series.
        """
        return self.__data_collection

    def as_tsfresh(self, long_format: pd.DataFrame) -> pd.DataFrame:
        """Transform the dataset into tsfresh features.

        This method may take a while depending on the size of the data.

        Parameters
        ----------
        long_format : pd.DataFrame
            Long-format dataframe with all time series to extract.

        Returns
        -------
        pandas.DataFrame
        """
        if self.tsfresh_features is None:
            extracted_features = extract_features(
                long_format, column_id='id', column_sort='time'
            )
            self.tsfresh_features = extracted_features

        return self.tsfresh_features

    def pad_collection(
        collection: Dict[str, List[np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """
        Pad the given collection with zeros in order to have the same length in
        all series.

        Parameters
        ----------
        collection : dict
            Dictionary whose keys are the star type and whose values are a list
            of numpy.array sequences representing the flux.

        Returns
        -------
        dict :
            Dictionary whose keys are the star type and whose values are
            numpy.array sequences representing the flux.
        """     
        raise NotImplementedError

    def as_lightkurve(self, collection: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Configure the dataset as a dictionary of 'lightkurve.TessLightCurve'
        objects.
        """
        raise NotImplementedError

    def as_dataframe(self, collection: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Configure the dataset as a pandas.DataFrame in long format.
        """
        if self.__id_dict is None:
            msg = "'get_ids' needs to be called first."
            raise ValueError(msg)

        df_dict = {}
        for key in collection:
            df_dict[key] = []
            for i, seq in enumerate(collection[key]):
                df = pd.DataFrame(seq, columns=['time', 'flux', 'flux_error']) # TODO: get from conventions
                df['id'] = self.__id_dict[key][i].replace('.txt', '')
                df['type'] = key
                
                df_dict[key].append(df)
                
            df_dict[key] = pd.concat(df_dict[key])
            
        df_long = pd.concat(df_dict.values())
        return df_long