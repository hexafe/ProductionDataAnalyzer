import pandas as pd
import dask.dataframe as dd
from typing import Optional, List, Union

def post_process_data(
    df: Union[pd.DataFrame, dd.DataFrame],
    date_col: Optional[str] = None,
    id_cols: Optional[List[str]] = None,
    numeric_threshold: float = 0.95,
    categorical_threshold: float = 0.1
) -> Union[pd.DataFrame, dd.DataFrame]:
    """
    Perform post-load data processing including:
    - Duplicate removal
    - Date parsing and sorting
    - Memory optimization through dtype conversion
    - ID column preservation

    Args:
        df (pd.DataFrame): Raw input DataFrame
        date_col (Optional[str], optional): Name of datetime column for parsing/sorting. Defaults to None.
        id_cols (Optional[List[str]], optional): Columns to preserve as strings. Defaults to None.
        numeric_threshold (float, optional): Proportion of numeric values needed for numeric conversion. Defaults to 0.95.
        categorical_threshold (float, optional): Proportion of unique values needed for categorical conversion. Defaults to 0.1.

    Returns:
        Union[pd.DataFrame, dd.DataFrame]: Processed DataFrame with optimized memory usage
        
    Raises:
        ValueError: For invalid threshold values or failed date parsing
    """
    df = clean_data(df, date_col)
    df = optimize_dtypes(df, date_col, id_cols, numeric_threshold, categorical_threshold)
    return df

def clean_data(
    df: Union[pd.DataFrame, dd.DataFrame],
    date_col: Optional[str]
) -> Union[pd.DataFrame, dd.DataFrame]:
    """
    Clean data by removing duplicates and parsing dates
    
    Args:
        df (Union[pd.DataFrame, dd.DataFrame]): Input data
        date_col (Optional[str]): Datetime column name
        
    Returns:
        Union[pd.DataFrame, dd.DataFrame]: Cleaned data
        
    Raises:
        ValueError: If date parsing fails completely
    """
    df = df.drop_duplicates(keep='first')
    
    if date_col and date_col in df.columns:
        df[date_col] = dd.to_datetime(df[date_col]) if isinstance(df, dd.DataFrame) else pd.to_datetime(df[date_col])
        if df[date_col].isnull().all().compute() if isinstance(df, dd.DataFrame) else df[date_col].isnull().all():
            raise ValueError(f"Failed to parse datetime column: {date_col}")
        df = df.sort_values(date_col)
        
    return df

def optimize_dtypes(
    df: Union[pd.DataFrame, dd.DataFrame],
    date_col: Optional[str],
    id_cols: Optional[List[str]],
    numeric_threshold: float,
    categorical_threshold: float
) -> Union[pd.DataFrame, dd.DataFrame]:
    """
    Optimize DataFrame memory usage while preserving ID columns
    
    Args:
        df (Union[pd.DataFrame, dd.DataFrame]): Input data
        date_col (Optional[str]): Datetime column name
        id_cols (Optional[List[str]]): Columns to preserve as strings
        numeric_threshold (float): Numeric conversion threshold
        categorical_threshold (float): Categorical conversion threshold
        
    Returns:
        Union[pd.DataFrame, dd.DataFrame]: Optimized data
        
    Raises:
        ValueError: For invalid threshold values
    """
    if not (0 <= numeric_threshold <= 1 and 0 <= categorical_threshold <= 1):
        raise ValueError("Thresholds must be between 0 and 1")
    
    id_cols = id_cols or []
    process_cols = [col for col in df.columns 
                   if col not in id_cols and col != date_col]
    
    if isinstance(df, dd.DataFrame):
        meta = df._meta
        return df.map_partitions(
            _optimize_partition,
            date_col=date_col,
            id_cols=id_cols,
            process_cols=process_cols,
            numeric_threshold=numeric_threshold,
            categorical_threshold=categorical_threshold,
            meta=meta
        )
    
    return _optimize_partition(
        df,
        date_col=date_col,
        id_cols=id_cols,
        process_cols=process_cols,
        numeric_threshold=numeric_threshold,
        categorical_threshold=categorical_threshold
    )

def _optimize_partition(
    df: pd.DataFrame,
    date_col: Optional[str],
    id_cols: List[str],
    process_cols: List[str],
    numeric_threshold: float,
    categorical_threshold: float
) -> pd.DataFrame:
    """Pandas-specific dtype optimization implementation"""
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
    
    # Ensure ID columns remain strings
    for col in id_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    return df
