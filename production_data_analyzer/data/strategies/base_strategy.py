from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
import dask.dataframe as dd
from typing import Union

class BaseLoader(ABC):
    """Abstract base class for data loading strategies"""
    supported_extensions: list[str] = []
    
    @abstractmethod
    def load(self, file_path: Union[str, Path]) -> Union[pd.DataFrame, dd.DataFrame]:
        """Load data from file path"""
        pass
    
    @classmethod
    def supports(cls, file_path: Union[str, Path]) -> bool:
        """Check if file is supported by the strategy"""
        return any(Path(file_path).suffix.lower() == ext for ext in cls.supported_extensions)
