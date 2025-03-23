import pandas as pd
from .base_strategy import BaseLoader

class ExcelHandler(BaseLoader):
    supported_extensions = ['.xls', '.xlsx']
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        
    def load(self, file_path: str) -> pd.DataFrame:
        return pd.read_excel(file_path, **self.kwargs)
