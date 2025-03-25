from typing import Union, Optional, List
from pathlib import Path
import pandas as pd
import warnings
import dask.dataframe as dd
from .strategies.csv_handler import CSVHandler
from .strategies.excel_handler import ExcelHandler
from .strategies.archive_handler import ArchiveHandler
from ..core.processing import DataProcessor
from .adapters import get_adapter

class DataLoader:
    """
    Hybrid data loading facade with ID column preservation
    
    Args:
        adapter (str): Execution environment ('colab' or 'local')
        use_dask (bool): Force Dask usage
        auto_switch_threshold (float): Memory ratio for auto-switch
        id_cols (Optional[List[str]]): Columns to preserve as strings
        **kwargs: File format parameters
    """
    def __init__(
        self,
        adapter: str = None,
        use_dask: bool = False,
        auto_switch_threshold: float = 0.6,
        id_cols: Optional[List[str]] = None,
        **kwargs
    ):
        if adapter is not None:
            warnings.warn(
                "'adapter' parameter is deprecated - environment detection is automatic", 
                DeprecationWarning,
                stacklevel=2
            )
        self.adapter = get_adapter()
        self.use_dask = use_dask
        self.auto_switch_threshold = auto_switch_threshold
        self.id_cols = id_cols or []
        self.csv_kwargs = kwargs.get('csv_kwargs', {})
        self.excel_kwargs = kwargs.get('excel_kwargs', {})
        
        # Initialize strategies with Dask support
        self.strategies = [
            CSVHandler(use_dask=use_dask, **self.csv_kwargs),
            ExcelHandler(use_dask=use_dask, **self.excel_kwargs),
            ArchiveHandler([CSVHandler(), ExcelHandler()])
        ]

    def load(
        self,
        source: Union[str, Path],
        date_col: Optional[str] = None
    ) -> Union[pd.DataFrame, dd.DataFrame]:
        """
        Load and process data with ID column preservation
        
        Args:
            source (Union[str, Path]): Data source
            date_col (Optional[str]): Datetime column name
            
        Returns:
            Union[pd.DataFrame, dd.DataFrame]: Processed data
            
        Raises:
            ValueError: For unsupported file types
        """
        raw_data = self._load_raw(source)
        
        # Auto-switch to Dask if needed
        if not self.use_dask and self._should_use_dask(raw_data):
            self.use_dask = True
            
        processed_data = DataProcessor.process(
            raw_data,
            date_col=date_col,
            id_cols=self.id_cols,
            numeric_threshold=0.95,
            categorical_threshold=0.1
        )
        
        return self._finalize_output(processed_data)

    def _should_use_dask(self, df: Union[pd.DataFrame, dd.DataFrame]) -> bool:
        """Check if Dask should be used based on memory"""
        if isinstance(df, dd.DataFrame):
            return True
            
        mem_usage = df.memory_usage(deep=True).sum() / 1e9  # GB
        total_mem = self.adapter.get_available_memory()
        return (mem_usage / total_mem) > self.auto_switch_threshold

    def _finalize_output(
        self,
        data: Union[pd.DataFrame, dd.DataFrame]
    ) -> Union[pd.DataFrame, dd.DataFrame]:
        """Convert to Pandas if possible"""
        if isinstance(data, dd.DataFrame) and self._can_convert_to_pandas(data):
            return data.compute()
        return data
