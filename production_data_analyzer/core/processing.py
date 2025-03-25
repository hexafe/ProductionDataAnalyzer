import pandas as pd
import dask.dataframe as dd
from typing import Optional, List, Union

class DataProcessor:
    def __init__(self):
        self.pipeline = [
            self._clean_data,
            self._optimize_dtypes
        ]
    
    def process(
        self,
        df: Union[pd.DataFrame, dd.DataFrame],
        date_col: Optional[str] = None,
        id_cols: Optional[List[str]] = None,
        numeric_threshold: float = 0.95,
        categorical_threshold: float = 0.1
    ) -> Union[pd.DataFrame, dd.DataFrame]:
        """
        Full processing pipeline with Dask support and threshold parameters
        
        Args:
            df: Input data (pandas or Dask)
            date_col: Datetime column name
            id_cols: Columns to preserve as strings
            numeric_threshold: Conversion threshold for numeric types
            categorical_threshold: Conversion threshold for categorical types
            
        Returns:
            Processed DataFrame (same type as input)
        """
        if not (0 <= numeric_threshold <= 1 and 0 <= categorical_threshold <= 1):
            raise ValueError("Thresholds must be between 0 and 1")
            
        context = {
            'date_col': date_col,
            'id_cols': id_cols or [],
            'numeric_threshold': numeric_threshold,
            'categorical_threshold': categorical_threshold
        }
        
        for step in self.pipeline:
            df = step(df, **context)
            
        return df
    
    def _clean_data(
        self,
        df: Union[pd.DataFrame, dd.DataFrame],
        date_col: Optional[str],
        **kwargs
    ) -> Union[pd.DataFrame, dd.DataFrame]:
        """Handle duplicates and datetime parsing"""
        df = df.drop_duplicates(keep='first')
        
        if date_col and date_col in df.columns:
            # Handle both pandas and Dask DataFrames
            dt_accessor = dd.to_datetime if isinstance(df, dd.DataFrame) else pd.to_datetime
            df[date_col] = dt_accessor(df[date_col], errors='coerce')
            
            # Validation check
            if isinstance(df, dd.DataFrame):
                invalid = df[date_col].isnull().all().compute()
            else:
                invalid = df[date_col].isnull().all()
                
            if invalid:
                raise ValueError(f"Failed to parse datetime column: {date_col}")
                
            df = df.sort_values(date_col)
            
        return df
    
    def _optimize_dtypes(
        self,
        df: Union[pd.DataFrame, dd.DataFrame],
        date_col: Optional[str],
        id_cols: List[str],
        numeric_threshold: float,
        categorical_threshold: float,
        **kwargs
    ) -> Union[pd.DataFrame, dd.DataFrame]:
        """Memory optimization with threshold-based type conversion"""
        process_cols = [col for col in df.columns 
                       if col not in id_cols and col != date_col]
        
        if isinstance(df, dd.DataFrame):
            meta = df._meta
            return df.map_partitions(
                self._optimize_partition,
                date_col=date_col,
                id_cols=id_cols,
                process_cols=process_cols,
                numeric_threshold=numeric_threshold,
                categorical_threshold=categorical_threshold,
                meta=meta
            )
        
        return self._optimize_partition(
            df,
            date_col=date_col,
            id_cols=id_cols,
            process_cols=process_cols,
            numeric_threshold=numeric_threshold,
            categorical_threshold=categorical_threshold
        )
    
    @staticmethod
    def _optimize_partition(
        df: pd.DataFrame,
        date_col: Optional[str],
        id_cols: List[str],
        process_cols: List[str],
        numeric_threshold: float,
        categorical_threshold: float
    ) -> pd.DataFrame:
        """Actual dtype optimization implementation for pandas"""
        for col in process_cols:
            if pd.api.types.is_string_dtype(df[col]):
                try_numeric = pd.to_numeric(df[col], errors='coerce')
                valid_ratio = try_numeric.notnull().mean()
                
                if valid_ratio >= numeric_threshold:
                    if try_numeric.dropna().apply(float.is_integer).all():
                        df[col] = pd.to_numeric(try_numeric, downcast='integer')
                    else:
                        df[col] = pd.to_numeric(try_numeric, downcast='float')
                else:
                    unique_ratio = df[col].nunique() / len(df)
                    if unique_ratio <= categorical_threshold:
                        df[col] = df[col].astype('category')
            elif pd.api.types.is_numeric_dtype(df[col]):
                if pd.api.types.is_integer_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], downcast='integer')
                else:
                    df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Preserve ID columns as strings
        for col in id_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        return df

def post_process_data(
    df: Union[pd.DataFrame, dd.DataFrame],
    date_col: Optional[str] = None,
    id_cols: Optional[List[str]] = None,
    numeric_threshold: float = 0.95,
    categorical_threshold: float = 0.1
) -> Union[pd.DataFrame, dd.DataFrame]:
    """Public interface for data processing pipeline"""
    processor = DataProcessor()
    return processor.process(
        df=df,
        date_col=date_col,
        id_cols=id_cols,
        numeric_threshold=numeric_threshold,
        categorical_threshold=categorical_threshold
    )
