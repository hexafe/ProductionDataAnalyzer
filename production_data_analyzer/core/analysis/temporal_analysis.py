import pandas as pd

class DataFilter:
    @staticmethod
    def filter_by_id(
        data: pd.DataFrame,
        id_column: str,
        valid_ids: list
    ) -> pd.DataFrame:
        if id_column not in data.columns:
            raise ValueError(f"ID column: '{id_column}' not found")
        
        data = data.copy()
        data[id_column] = data[id_column].astype(str)
        valid_ids = [str(id) for id in valid_ids]
        
        return data[data[id_column].isin(valid_ids)]
