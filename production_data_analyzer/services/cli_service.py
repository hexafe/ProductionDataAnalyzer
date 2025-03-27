from pydantic import BaseModel
import pandas as pd
from pathlib import Path
import tempfile
from datetime import datetime, timedelta
import shutil
from typing import Optional, Dict
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

class SessionConfig(BaseModel):
    retention_days: int = 7
    default_format: str = 'parquet'

class CLIConfig(BaseModel):
    loader: DataLoaderConfig = DataLoaderConfig()
    formatter: FormatterConfig = FormatterConfig()
    sessions: SessionConfig = SessionConfig()

class CLIService:
    def __init__(self, config: CLIConfig):
        # Existing initialization
        self.loader = DataLoader(
            adapter=config.loader.adapter,
            use_dask=config.loader.use_dask,
            auto_switch_threshold=config.loader.auto_switch_threshold
        )
        self.formatter = OutputFormatter(config.formatter)
        
        # New session management
        self.temp_root = Path(tempfile.gettempdir()) / "production_analyzer"
        self.temp_root.mkdir(exist_ok=True)
        self.active_sessions: Dict[str, pd.DataFrame] = {'default': None}
        self.current_session = 'default'
        self.session_config = config.sessions
        
        # Cleanup old sessions
        self._clean_old_sessions()

    # Existing methods (modified for session support)
    def load_data(self, source: str, session: str = None) -> str:
        """Modified to support sessions"""
        session = session or self.current_session
        self.active_sessions[session] = self.loader.load(source)
        return f"Data loaded into session '{session}'\n" + \
            self.formatter.format_table(self.active_sessions[session].head())

    def analyze_temporal(self, time_col: str, window: str = '7D', session: str = None) -> str:
        """Modified to support sessions"""
        session = session or self.current_session
        df = self.active_sessions.get(session)
        
        if df is None:
            raise ValueError(f"No data loaded in session '{session}'")
            
        analyzer = TemporalAnalyzer(df, time_col)
        result = analyzer.rolling_aggregation(window)
        return self.formatter.format_table(result)

    # New session methods
    def save_session(self, session_name: str) -> Path:
        session_dir = self.temp_root / session_name
        session_dir.mkdir(exist_ok=True)
        
        if df := self.active_sessions.get(session_name):
            df.to_parquet(session_dir / "data.parquet")
        return session_dir

    def load_session(self, session_name: str) -> bool:
        session_dir = self.temp_root / session_name
        data_file = session_dir / "data.parquet"
        
        if data_file.exists():
            self.active_sessions[session_name] = pd.read_parquet(data_file)
            return True
        return False
    
    def delete_session(self, session_name: str) -> bool:
        session_dir = self.temp_root / session_name
        if session_dir.exists():
            shutil.rmtree(session_dir)
            return True
        return False

    def switch_session(self, session_name: str) -> bool:
        if session_name in self.active_sessions:
            self.current_session = session_name
            return True
        return False

    # Maintain backward compatibility
    @property
    def current_data(self) -> Optional[pd.DataFrame]:
        """Proxy to current session's data"""
        return self.active_sessions.get(self.current_session)

    @current_data.setter
    def current_data(self, value: pd.DataFrame):
        """Proxy to current session's data"""
        self.active_sessions[self.current_session] = value

    # Existing class method
    @classmethod
    def load_config(cls) -> CLIConfig:
        return CLIConfig()

    # Private methods
    def _clean_old_sessions(self):
        cutoff = datetime.now() - timedelta(days=self.session_config.retention_days)
        for session_dir in self.temp_root.iterdir():
            if session_dir.is_dir():
                mtime = datetime.fromtimestamp(session_dir.stat().st_mtime)
                if mtime < cutoff:
                    shutil.rmtree(session_dir)
                    
    def get_memory_usage(self):
        if self.current_data is not None:
            return round(self.current_data.memory_usage(deep=True).sum() / 1e6, 2)  # MB
        return 0.0
    
    def list_sessions(self):
        return [d.name for d in self.temp_root.iterdir() if d.is_dir()]
