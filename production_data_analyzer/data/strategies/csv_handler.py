import pandas as pd
import dask.dataframe as dd
from pathlib import Path
from typing import Union, Tuple
import chardet
import csv
from io import StringIO
from .base_strategy import BaseLoader
from ..adapters import detect_environment, get_adapter

class CSVHandler(BaseLoader):
    """CSV file loader with auto-detection of format parameters and Colab support
    
    Features:
    - Automatic detection of separator, decimal point, and encoding
    - Fallback to European defaults (';' ',' 'utf-8')
    - Support for both local files and Google Colab uploads
    - Memory optimization through categorical conversion
    - Pandas/Dask compatibility
    
    Example:
        >>> handler = CSVHandler()
        >>> df = handler.load("production_data.csv")  # Local file
        >>> colab_df = handler.load()  # Triggers upload in Colab
    """
    
    supported_extensions = ['.csv']
    
    def __init__(self, use_dask: bool = False, **kwargs):
        """Initialize CSV handler with loading preferences
        
        Args:
            use_dask (bool): Use Dask for parallel processing of large files.
                Default: False
            **kwargs: Additional pandas.read_csv/dask.read_csv parameters that
                override auto-detected values
                
        Example:
            >>> handler = CSVHandler(use_dask=True, nrows=1000)  # Load first 1000 rows
        """
        self.use_dask = use_dask
        self.defaults = {
            'sep': ';',
            'decimal': ',',
            'encoding': 'utf-8'
        }
        self.kwargs = {**self.defaults, **kwargs}
        self.adapter = get_adapter()

    def load(self, file_path: Union[str, Path] = None) -> Union[pd.DataFrame, dd.DataFrame]:
        """Load CSV data with environment-appropriate handling
        
        Args:
            file_path: Optional path for local files (ignored in Colab)
            
        Returns:
            Combined DataFrame from all processed files
            
        Raises:
            FileNotFoundError: For missing local files
            ValueError: For empty uploads or invalid CSVs
            
        Example (Local):
            >>> df = CSVHandler().load("data.csv")
            
        Example (Colab):
            >>> df = CSVHandler().load()  # Triggers file upload dialog
        """
        if detect_environment() == 'colab':
            if file_path is not None:
                print("Warning: file_path argument ignored in Colab environment")
            return self._colab_load()
            
        # Local file handling
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        return self._load_file(file_path)

    def _load_file(self, path: Path) -> Union[pd.DataFrame, dd.DataFrame]:
        """Internal method to load single CSV file with error handling.
        
        Args:
            path: Path to CSV file
            
        Returns:
            Loaded DataFrame (pandas or Dask)
            
        Raises:
            ValueError: For parsing errors
            UnicodeDecodeError: If encoding detection fails
        """
        detected_params = self._detect_csv_params(path)
        final_params = {**detected_params, **self.kwargs}
        
        try:
            return self._try_load(path, final_params)
        except Exception as e:
            print(f"Load failed with detected params: {e}. Trying defaults...")
            return self._try_load(path, self.defaults)

    def _try_load(self, path: Path, params: dict) -> Union[pd.DataFrame, dd.DataFrame]:
        """Attempt CSV loading with specified parameters.
        
        Args:
            path: Path to CSV file
            params: Dictionary of pandas.read_csv parameters
            
        Returns:
            Loaded DataFrame
            
        Raises:
            ValueError: For parsing errors or failed loading
        """
        try:
            if self.use_dask:
                return dd.read_csv(str(path), **params)
            return pd.read_csv(str(path), **params)
        except pd.errors.ParserError as e:
            raise ValueError(f"CSV parsing error in {path.name}: {str(e)}")
        except UnicodeDecodeError:
            return self._retry_with_fallback_encoding(path, params)
        except Exception as e:
            raise ValueError(f"Failed to load {path.name}: {str(e)}")

    def _retry_with_fallback_encoding(self, path: Path, params: dict):
        """Attempt loading with alternative encodings.
        
        Args:
            path: Path to CSV file
            params: Current loading parameters
            
        Returns:
            DataFrame loaded with alternative encoding
            
        Raises:
            ValueError: If no compatible encoding found
        """
        for encoding in ['latin-1', 'iso-8859-1', 'cp1252']:
            try:
                return pd.read_csv(str(path), **{**params, 'encoding': encoding})
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Failed to decode {path.name} with common encodings")

    def _detect_csv_params(self, path: Path) -> dict:
        """Auto-detect CSV format parameters from file sample.
        
        Args:
            path: Path to CSV file
            
        Returns:
            Dictionary with detected parameters:
            {
                'sep': detected separator,
                'decimal': detected decimal point,
                'encoding': detected encoding
            }
        """
        sample = path.read_bytes()[:4096]  # Analyze first 4KB
        
        encoding = self._detect_encoding(sample)
        sep, decimal = self._detect_separator_and_decimal(sample, encoding)
        
        return {
            'sep': sep,
            'decimal': decimal,
            'encoding': encoding
        }

    def _detect_encoding(self, sample: bytes) -> str:
        """Detect file encoding using statistical analysis.
        
        Args:
            sample: First 4096 bytes of the file
            
        Returns:
            Detected encoding string
            
        Note:
            Uses chardet library with confidence threshold > 0.9
        """
        result = chardet.detect(sample)
        return result['encoding'] if result['confidence'] > 0.9 else self.defaults['encoding']

    def _detect_separator_and_decimal(self, sample: bytes, encoding: str) -> Tuple[str, str]:
        """Detect separator and decimal using CSV pattern analysis.
        
        Args:
            sample: File bytes sample
            encoding: Detected encoding
            
        Returns:
            Tuple of (separator, decimal point)
        """
        try:
            text = sample.decode(encoding, errors='ignore')
            dialect = csv.Sniffer().sniff(text)
            sep = dialect.delimiter
            decimal = self._detect_decimal(text, sep)
            return sep, decimal
        except:
            return self.defaults['sep'], self.defaults['decimal']

    def _detect_decimal(self, text: str, sep: str) -> str:
        """Detect decimal separator through numeric pattern matching.
        
        Args:
            text: Decoded text sample
            sep: Detected column separator
            
        Returns:
            Most probable decimal separator (',' or '.')
        """
        reader = csv.reader(StringIO(text))
        counts = {',': 0, '.': 0}
        
        for row in reader:
            for value in row:
                if ',' in value and sep != ',': counts[','] += 1
                if '.' in value and sep != '.': counts['.'] += 1
                
        return max(counts, key=counts.get) or self.defaults['decimal']

    def _colab_load(self) -> Union[pd.DataFrame, dd.DataFrame]:
        """Handle Google Colab file upload and processing
        
        Returns:
            Combined DataFrame from uploaded files
            
        Raises:
            ValueError: For empty uploads or invalid files
            
        Note:
            - Triggers Colab file upload dialog
            - Processes all uploaded CSV files
            - Filenames are not filtered, only CSV extensions
        """
        uploaded_files = self.adapter.upload_files()
        if not uploaded_files:
            raise ValueError("No files uploaded in Colab session")
            
        dfs = []
        for file in uploaded_files:
            if file.suffix.lower() == '.csv':
                try:
                    dfs.append(self._load_file(file))
                except Exception as e:
                    print(f"Skipping {file.name}: {str(e)}")
                    continue
                    
        if not dfs:
            raise ValueError("No valid CSV files found in uploaded files")
            
        return self._merge_dataframes(dfs)

    def _merge_dataframes(self, dfs: list) -> Union[pd.DataFrame, dd.DataFrame]:
        """Merge multiple DataFrames with memory optimization.
        
        Args:
            dfs: List of DataFrames to merge
            
        Returns:
            Combined DataFrame with categorical conversion for
            low-cardinality columns (<10% unique values)
        """
        if self.use_dask:
            return dd.concat(dfs, axis=0) if len(dfs) > 1 else dfs[0]
            
        merged = pd.concat(dfs, ignore_index=True)
        return merged.astype({
            col: 'category' for col in merged.columns 
            if merged[col].nunique() / len(merged) < 0.1
        })
