import pandas as pd
from typing import Optional

class TemporalAnalyzer:
    """
    Time-based production data analysis toolkit
    """
    
    def __init__(self, df: pd.DataFrame, time_col: str):
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            raise ValueError(f"Column '{time_col}' must be datetime type")
        self.df = df.sort_values(time_col)
        self.time_col = time_col

    def rolling_aggregation(
        self,
        window: str = '7D',
        metrics: dict = None
    ) -> pd.DataFrame:
        """
        Calculate rolling window statistics
        
        Args:
            window: Offset string representing window size
            metrics: Dictionary of {column: [stat functions]}
            
        Returns:
            DataFrame with rolling statistics
        """
        default_metrics = {
            'mean': pd.Series.mean,
            'std': pd.Series.std,
            'min': pd.Series.min,
            'max': pd.Series.max
        }
        metrics = metrics or default_metrics
        
        return self.df.set_index(self.time_col).rolling(window).agg(metrics)

    def time_based_filter(
        self,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        """
        Filter data within specified time range
        
        Args:
            start: Start timestamp (inclusive)
            end: End timestamp (exclusive)
            
        Returns:
            Filtered DataFrame
        """
        mask = pd.Series(True, index=self.df.index)
        if start:
            mask &= (self.df[self.time_col] >= start)
        if end:
            mask &= (self.df[self.time_col] < end)
            
        return self.df[mask].copy()
