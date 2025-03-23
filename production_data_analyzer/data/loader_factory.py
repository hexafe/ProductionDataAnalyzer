from .strategies.csv_handler import CSVHandler
from .strategies.excel_handler import ExcelHandler
from ..loader import BaseLoader

class DataLoaderFactory:
    @staticmethod
    def get_loader(file_type: str, **kwargs) -> BaseLoader:
        if file_type == 'csv':
            return CSVHandler(**kwargs)
        elif file_type == 'excel':
            return ExcelHandler(**kwargs)
        
        raise ValueError(f"Unsupported format: {file_type}")
