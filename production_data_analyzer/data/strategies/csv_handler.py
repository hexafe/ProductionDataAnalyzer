import pandas as pd
from ..loader import BaseLoader

class CSVHandler(BaseLoader):
    def __init__(self, date_col: str = None, id_cols: list = None):
        self.date_col = date_col
        self.id_cols = id_cols or []
        
    def load(self, file_path: str) -> pd.DataFrame:
        return pd.read_csv(
            file_path,
            sep=';',
            decimal=',',
            parse_dates=[self.date_col] if self.date_col else False,
            dtype={col: str for col in self.id_cols}
        )
