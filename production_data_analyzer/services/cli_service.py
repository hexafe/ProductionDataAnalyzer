from pydantic import BaseModel
import pandas as pd
from typing import Optional
from ..core import ProductionDataFilter, TemporalAnalyzer
from ..data import DataLoader
from ..cli.utils.output_formatters import OutputFormatter

class DataLoaderConfig(BaseModel):
    adapter: str = 'auto'
    use_dask: bool = False
    auto_switch_threshold: float = 0.6

class FormatterConfig(BaseModel):
    output_format: str = 'table'
    table_style: str = 'psql'

class CLIConfig(BaseModel):
    loader: DataLoaderConfig = DataLoaderConfig()
    formatter: FormatterConfig = FormatterConfig()

class CLIService:
    def __init__(self, config: CLIConfig):
        self.loader = DataLoader(
            adapter=config.loader.adapter,
            use_dask=config.loader.use_dask,
            auto_switch_threshold=config.loader.auto_switch_threshold
        )
        self.formatter = OutputFormatter(config.formatter)
        self.current_data: Optional[pd.DataFrame] = None

    def load_data(self, source: str) -> str:
        self.current_data = self.loader.load(source)
        return self.formatter.format_table(self.current_data.head())

    def analyze_temporal(self, time_col: str, window: str = '7D') -> str:
        if self.current_data is None:
            raise ValueError("No data loaded. Use load-data first")
            
        analyzer = TemporalAnalyzer(self.current_data, time_col)
        result = analyzer.rolling_aggregation(window)
        return self.formatter.format_table(result)

    @classmethod
    def load_config(cls) -> CLIConfig:
        return CLIConfig()
