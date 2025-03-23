import pandas as pd
import dask.dataframe as dd
from pathlib import Path
from typing import Union
from .base_strategy import BaseLoader
from ..adapters import detect_environment, get_adapter

class CSVHandler(BaseLoader):
    """CSV file loading strategy with automatic Colab support"""
    supported_extensions = ['.csv']
    
    def __init__(self, use_dask: bool = False, **kwargs):
        self.use_dask = use_dask
        self.kwargs = kwargs
        self.adapter = get_adapter()

    def load(self, file_path: Union[str, Path]) -> Union[pd.DataFrame, dd.DataFrame]:
        if detect_environment() == 'colab':
            return self._colab_load()
            
        if self.use_dask:
            return dd.read_csv(file_path, **self.kwargs)
        return pd.read_csv(file_path, **self.kwargs)

    def _colab_load(self) -> Union[pd.DataFrame, dd.DataFrame]:
        """Colab-specific file upload handling"""
        uploaded = self.adapter.upload_files()
        dfs = []
        for file in uploaded:
            if file.suffix.lower() in self.supported_extensions:
                if self.use_dask:
                    dfs.append(dd.read_csv(file, **self.kwargs))
                else:
                    dfs.append(pd.read_csv(file, **self.kwargs))
        return pd.concat(dfs) if dfs else pd.DataFrame()
