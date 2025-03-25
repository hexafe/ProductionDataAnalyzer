from pathlib import Path
from typing import List, Union
import pandas as pd
import dask.dataframe as dd
from .base_strategy import BaseLoader
from ..adapters import get_adapter

class ArchiveHandler(BaseLoader):
    """Archive loading strategy with recursive processing"""
    supported_extensions = ['.zip', '.7z', '.rar', '.tar', '.gz']
    
    def __init__(self, strategies: List[BaseLoader]):
        self.adapter = get_adapter()
        self.strategies = strategies

    def load(self, file_path: Path) -> Union[pd.DataFrame, dd.DataFrame]:
        """Loac and combine data from archive"""
        extract_dir = self.adapter.extract_archive(file_path)
        processed_files = self._process_extracted_files(extract_dir)
        
        return self._combine_results(processed_files)
    
    def _process_extracted_files(self, directory: Path) -> list:
        processed = []
        for file in directory.rglob('*'):
            for strategy in self.strategies:
                if strategy.supports(file):
                    processed.append({strategy.load(file)})
                    break
        return processed
    
    def _combine_results(self, results: list) -> Union[pd.DataFrame, dd.DataFrame]:
        if not results:
            raise ValueError("No loadable files found in archive")
        
        if isinstance(results[0], pd.DataFrame):
            return pd.concat(results, ignore_index=True)
        return dd.concat(results, axis=0)
