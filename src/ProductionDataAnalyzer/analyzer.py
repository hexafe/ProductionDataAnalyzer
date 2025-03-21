from typing import Union, Dict, List, Tuple, Optional, Any
import os
import io
import sys
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.figure_factory as ff
import panel as pn
import matplotlib.dates as mdates
import datetime
import gspread
from google.colab import auth
from google.auth import default
from pyunpack import Archive
import shutil
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.style import Style
from scipy.stats import kurtosis, shapiro
from IPython import get_ipython
from concurrent.futures import ThreadPoolExecutor, as_completed
import dask.dataframe as dd

class ProductionDataAnalyzer:
    """
    Industrial data analysis toolkit with integrated quality analytics

    **Google Colab Integration**:
    1. Clone repository: `!git clone https://github.com/hexafe/ProductionDataAnalyzer.git`
    2. Navigate to the project directory: `%cd ProductionDataAnalyzer`
    3. Install dependencies: `!pip install -r requirements.txt`
    4. Import directly: `from ProductionDataAnalyzer import ProductionDataAnalyzer`
    !git clone https://github.com/hexafe/ProductionDataAnalyzer.git
    %cd ProductionDataAnalyzer
    !pip install -r requirements.txt
    from ProductionDataAnalyzer import ProductionDataAnalyzer

    Core Capabilities - to be implemented:
    ------------------
    1. **Data Management**
       - Multi-format ingestion (CSV/Excel/ZIP/7z/RAR)
       - Time-indexed dataset organization
       - Part ID-based filtering and segmentation

    2. **Temporal Analysis**
       - Flexible aggregation (minute/hourly/daily)
       - Statistical timeframe comparison (T-test)
       - Interactive time-series visualization

    3. **Machine Learning & Advanced Analytics** - ambitious plan to be implemented
       - Defect prediction models:
         * Gradient Boosting (XGBoost/LightGBM/CatBoost)
         * Hybrid temporal models (LSTM-XGB ensembles)
         * Constrained optimization (process-aware ML)
       - Explainable AI:
         * SHAP value analysis with process parameter mapping
         * Counterfactual defect scenario generation
         * Root cause attribution clustering
       - Adaptive monitoring:
         * Concept drift detection (ADWIN/Page-Hinkley)
         * Automated hyperparameter tuning (Optuna)
         * Causal impact validation

    4. **Industrial Statistics**
       - Statistical Process Control (SPC) charting
       - Process capability analysis (Cp/Cpk/Ppk)
       - Multivariate correlation analysis

    Example Workflow:
    -----------------
    >>> analyzer = ProductionDataAnalyzer(
    ...     raw_data,
    ...     date_col='timestamp',
    ...     part_id_col='serial_number'
    ... )

    # Configure analysis parameters
    >>> analyzer.set_control_limits(
    ...     parameters={'pressure': (80, 120)},
    ...     spc_method='xbar_r'
    ... )

    # Train defect prediction model
    >>> model = analyzer.train_defect_model(
    ...     algorithm='constrained_xgb',
    ...     features=['temp', 'vibration', 'cycle_time'],
    ...     monotone_constraints={'temp': -1}
    ... )

    # Generate SHAP analysis report
    >>> shap_report = analyzer.generate_shap_summary(
    ...     model=model,
    ...     reference_period='2024-01',
    ...     comparison_period='2024-02'
    ... )

    # Monitor model performance
    >>> drift_status = analyzer.check_concept_drift(
    ...     model=model,
    ...     detector_type='adwin',
    ...     update_strategy='dynamic_retrain'
    ... )
    """

    IN_COLAB = 'google.colab' in sys.modules
    console = Console(force_terminal=True if not IN_COLAB else False)

    def __init__(
        self,
        production_data: pd.DataFrame,
        date_col: str = None,
        enable_rich: bool = True
    ):
        """
        Initialize a production data analyzer with raw data and configuration

        Parameters:
            production_data (pd.DataFrame): Input data containing production records
                                            Must contain a datetime column and various parameters to analyze
            date_col (str):                 Name of the datetime column used for temporal analysis
                                            Default: None - class can manage data without datetime column
            enable_rich (bool):             Enable rich terminal output (local only)
                                            Default: True

        Raises:
            TypeError:  If input data is not a pandas DataFrame
            ValueError: If specified datetime column is invalid

        Attributes:
            production (pd.DataFrame):  Reference to stored production data
            date_col (str):             Name of datetime column
            agg_data (pd.DataFrame):    Placeholder for aggregated analysis results
            param_limits (dict):        Storage for parameter validation limits {param: (min, max)}
            selected_params (list):     Parameters selected for analysis (all columns by default)
            agg_freq (str):             Time frequency for aggregation ('D' = daily)
            date_formatters (dict):     Matplotlib date formatters for different time resolutions

        Example:
            >>> analyzer = ProductionDataAnalyzer(production_data=df, date_col='timestamp')
            Analyzer initialized with 1000 records (12 parameters)
        """
        # Input validation
        if not isinstance(production_data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")

        # Check if date_col exist and is in proper type
        if date_col:
            if date_col not in production_data.columns:
                raise ValueError(f"Column '{date_col}' not found in input data")
            if production_data[date_col].isna().all():
                raise ValueError(f"All values in date column '{date_col}' are null")
            if not pd.api.types.is_datetime64_any_dtype(production_data[date_col]):
                raise ValueError(f"Column '{date_col}' must be datetime type")

        # Core attributes
        self.date_col = date_col
        self.production = production_data.copy()
        self.agg_data = pd.DataFrame()
        self.param_limits = {}

        # Configuration
        self.selected_params = list(production_data.columns)
        self.agg_freq = 'D'
        self.date_formatters = {
            'day': mdates.DateFormatter('%Y-%m-%d'),
            'week': mdates.DateFormatter('%Y-W%U'),
            'month': mdates.DateFormatter('%Y-%m'),
            'year': mdates.DateFormatter('%Y')
        }

        self.enable_rich = enable_rich if not self.IN_COLAB else False
        if self.enable_rich:
            self.console = Console()
            self.console.print(
                Markdown("# Production Data Analyzer initialized"),
                style="bold blue"
            )

        # Initialization feedback
        print(f"Analyzer initialized with {len(self.production):,} records ({len(self.selected_params)} parameters)")

    def _print(self, message, style=None):
        if self.enable_rich:
            self.console.print(message, style=style)
        else:
            print(message)

    @staticmethod
    def _smart_round(value, decimal_places: int = 4):
        """Format the number in fixed-point notation"""
        if isinstance(value, (int, np.integer)):
            return int(value)
        elif isinstance(value, (float, np.floating)):
            value = float(value)
            s = f"{value:.15f}".rstrip('0').rstrip('.')
            if '.' in s:
                _, decimal_part = s.split('.')
                if len(decimal_part) > decimal_places:
                    value = round(value, decimal_places)
            return int(value) if value.is_integer() else value
        else:
            return value

    @staticmethod
    def _datetime_converter(date_str):
        """
        Convert an input date value to a pandas Timestamp using multiple fallback parsing strategies

        Parameters:
            date_str (str, pd.Timestamp, datetime.datetime): The date value to convert
                This can be a string in various common date formats, a pandas Timestamp or a datetime.datetime object

        Returns:
            pd.Timestamp if successfully parsed datetime object
            pd.NaT if:
                - input is None/NaN
                - all format conversion attempts fail
                - non-parsable non-datetime input

        Examples:
            Valid input formats include:
                - '2023-07-17 14:30:45.123'
                - '17.07.2023 14:30' (European format)
                - '7/17/2023 02:30 PM' (US format)
                - '20230717143045' (Compact notation)
                - pandas.Timestamp objects pass through unchanged

        Raises:
            pd.errors.OutOfBoundsDatetime: If parsed date exceeds pandas' timestamp range
            Note: Most errors return NaT rather than raising exception
        """
        if isinstance(date_str, (pd.Timestamp, datetime.datetime)):
            return date_str

        formats = [
            # ISO variants
            '%Y-%m-%d %H:%M:%S.%f',    # 2023-07-17 14:30:45.123
            '%Y-%m-%dT%H:%M:%S.%fZ',   # ISO with timezone
            '%Y%m%d%H%M%S',            # 20230717143045
            
            # European-style with various separators
            '%d.%m.%Y %H:%M:%S',       # 17.07.2023 14:30:45
            '%d/%m/%Y %H:%M',          # 17/07/2023 14:30
            
            # US-style formats
            '%m/%d/%Y %I:%M %p',       # 7/17/2023 02:30 PM
            '%b %d, %Y %H:%M',         # Jul 17, 2023 14:30
            
            # Fallback formats
            '%Y-%m-%d',                # Date-only
            '%H:%M:%S %d-%b-%Y'        # 14:30:45 17-Jul-2023
        ]

        for fmt in formats:
            try:
                return pd.to_datetime(date_str, format=fmt, exact=True)
            except:
                continue
                
        try:
            return pd.to_datetime(date_str, dayfirst=True, errors='coerce')
        except:
            return pd.NaT
        
    @staticmethod
    def upload_files(
        date_col: str = None,
        id_cols: List[str] = None,
        csv_kwargs: dict = None,
        excel_kwargs: dict = None,
        archive_ext: tuple = ('.zip', '.7z', '.rar', '.tar', '.gz'),
        csv_ext: tuple = ('.csv',),
        excel_ext: tuple = ('.xls', '.xlsx'),
        tmp_dir: str = None,
        remove_tmp_dir: bool = True,
        chunksize: int = 100000,
        local_files: List[str] = None,
        max_workers: int = 4
    ) -> pd.DataFrame:
        """
        Upload/read files (CSV, Excel, and various archives) from Google Colab/local environment and return a combined DataFrame
        Processes CSV, Excel, and compressed archives with parallel processing and memory optimization

        Custom keyword arguments for reading CSV and Excel files can be provided via
        `csv_kwargs` and `excel_kwargs`, respectively. Defaults are used if not specified

        Parameters:
            date_col (str):             Name of the column to parse as datetime
            id_cols (List[str]):        List of names for ID columns to preserve as strings
            csv_kwargs (dict):          Custom parameters for pd.read_csv(). Merged with defaults:
                                        {
                                            'sep': ';', 
                                            'decimal': ',',
                                            'parse_dates': [date_col] if date_col else False,
                                            'dayfirst': True,
                                            'na_values': ['\\N', ''],
                                            'keep_default_na': False
                                        }
                                        Default: None
            excel_kwargs (dict):        Custom parameters for pd.read_excel(). Merged with:
                                        {'engine': 'openpyxl', 'na_values': ['\\N', '']}
                                        Default: None
            archive_ext (tuple):        File extensions recognized as archives
                                        Default: ('.zip', '.7z', '.rar', '.tar', '.gz')
            csv_ext (tuple):            File extensions recognized as CSV files
                                        Default: ('.csv',)
            excel_ext (tuple):          File extensions recognized as Excel files
                                        Default: ('.xls', '.xlsx')
            tmp_dir (str):              Temporary directory path for file processing. Created automatically if None
                                        Default: None
            remove_tmp_dir (bool):      Whether to remove the temporary directory after processing. Recommended for security
                                        Default: True
            chunksize (int):            Number of rows per chunk for processing large CSV files
                                        If None, CSV files are read in one go
                                        Default: 100000
            local_files (List[str]):    List of local files path to load in local env
                                        Default: None
            max_workers (int):          Parallel processing threads for file reading
                                        Default: 4

        Environment-specific behavior:
            Colab:
                - Uses interactive file upload widget
            Local:
                - Requires explicit file paths in local_files
                - Handles system file paths directly

        Returns:
            pd.DataFrame: Combined DataFrame after processing all files

        Raises:
            ValueError:         Invalid date_col specification or missing local_files in local environment
            FileNotFoundError:  Missing local_files paths in local mode
            RuntimeError:       Unsupported file formats or processing failures
            PermissionError:    File system access issues for tmp_dir
        """
        # Configure temporary directory
        tmp_dir_path = Path(tmp_dir) if tmp_dir else Path(
            tempfile.mkdtemp(prefix="prod_analysis_")
        )
        tmp_dir_path.mkdir(parents=True, exist_ok=True)

        # Configure file handlers
        csv_kwargs = ProductionDataAnalyzer._configure_csv_reader(
            date_col, id_cols, csv_kwargs)
        excel_kwargs = ProductionDataAnalyzer._configure_excel_reader(excel_kwargs)

        processed_files = set()
        dfs = []
        executor = None

        try:
            # Environment-specific file collection
            if ProductionDataAnalyzer.IN_COLAB:
                from google.colab import files
                uploaded = files.upload()
                for fn, content in uploaded.items():
                    file_path = tmp_dir_path / fn
                    file_path.write_bytes(content)
                    ProductionDataAnalyzer._process_archive(
                        file_path, tmp_dir_path, archive_ext)
            else:
                if not local_files:
                    raise ValueError("Local execution requires 'local_files' parameter")
                for file_path in map(Path, local_files):
                    if not file_path.exists():
                        raise FileNotFoundError(f"File not found: {file_path}")
                    ProductionDataAnalyzer._process_archive(
                        file_path, tmp_dir_path, archive_ext)

            # Collect all processable files
            processed_files = {
                f for f in tmp_dir_path.rglob('*')
                if f.is_file() and f.suffix.lower() in (csv_ext + excel_ext)
            }

            # Parallel processing with ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for file_path in processed_files:
                    futures.append(executor.submit(
                        ProductionDataAnalyzer._read_data_file,
                        file_path, csv_kwargs, excel_kwargs,
                        csv_ext, excel_ext, chunksize
                    ))
                
                for future in as_completed(futures):
                    try:
                        dfs.append(future.result())
                    except Exception as e:
                        print(f"Error processing file: {str(e)}")
                        raise

        except Exception as e:
            raise RuntimeError(f"File processing failed: {str(e)}") from e
        finally:
            if remove_tmp_dir:
                shutil.rmtree(tmp_dir_path, ignore_errors=True)

        if not dfs:
            raise RuntimeError(
                "No valid data processed - check:\n"
                "1. Supported formats (CSV, Excel, ZIP, 7z, RAR)\n"
                "2. Archive contents\n"
                "3. Column names match expectations"
            )

        combined = pd.concat(dfs, ignore_index=True, copy=False)
        return ProductionDataAnalyzer._post_merge_cleanup(combined, date_col, id_cols)

    @staticmethod
    def _configure_csv_reader(
        date_col: str | None, 
        id_cols: List[str] | None, 
        csv_kwargs: dict | None
    ) -> dict:
        """
        Configures CSV reader parameters with safety defaults

        Parameters:
            date_col (str | None):       datetime column name for parsing
            id_cols (List[str] | None):  columns to preserve as strings
            csv_kwargs (dict | None):    user-provided CSV parameters

        Returns:
            dict: Merged configuration with priority:
                1. User-provided csv_kwargs
                2. ID column type preservation
                3. European-format defaults

        Raises:
            ValueError: If conflicting date parsing configuration
        """
        defaults = {
            'sep': ';',
            'decimal': ',',
            'parse_dates': [date_col] if date_col else False,
            'dayfirst': True,
            'na_values': ['\\N', ''],
            'keep_default_na': False,
            'dtype': {col: str for col in id_cols} if id_cols else None,
            'engine': 'c',
            'memory_map': True
        }
        return {**defaults, **(csv_kwargs or {})}

    @staticmethod
    def _configure_excel_reader(excel_kwargs: dict | None) -> dict:
        """
        Configures Excel reader with openpyxl engine

        Parameters:
            excel_kwargs (dict | None):  user-provided Excel parameters

        Returns:
            dict: Merged configuration ensuring:
                - openpyxl engine usage
                - Consistent NA value handling
                - Type inference optimization

        Raises:
            ImportError: If openpyxl not installed
        """
        defaults = {
            'engine': 'openpyxl',
            'na_values': ['\\N', ''],
            'keep_default_na': False
        }
        return {**defaults, **(excel_kwargs or {})}

    @staticmethod
    def _process_archive(file_path: Path, tmp_dir: Path, archive_ext: tuple) -> None:
        """
        Handles archive extraction with format validation

        Parameters:
            file_path (Path):            Path to archive file
            tmp_dir (Path):              Target directory for extraction
            archive_ext (tuple):         Valid archive extensions

        Raises:
            RuntimeError:                Unsupported archive format or extraction failure
            ValueError:                  Invalid archive structure
        """
        if file_path.suffix.lower() in archive_ext:
            try:
                Archive(str(file_path)).extractall(str(tmp_dir))
            except Exception as e:
                print(f"Failed to extract {file_path.name}: {str(e)}")
                if (tmp_dir / file_path.name).exists():
                    (tmp_dir / file_path.name).unlink()

    @staticmethod
    def _read_data_file(
        file_path: Path, 
        csv_kwargs: dict, 
        excel_kwargs: dict,
        csv_ext: tuple, 
        excel_ext: tuple, 
        chunksize: int | None
    ) -> pd.DataFrame:
        """
        Reads individual data file

        Parameters:
            file_path (Path):            File to process
            csv_kwargs (dict):           Configured CSV parameters
            excel_kwargs (dict):         Configured Excel parameters
            csv_ext (tuple):             Valid CSV extensions
            excel_ext (tuple):           Valid Excel extensions
            chunksize (int | None):      Chunk size for CSV processing

        Returns:
            pd.DataFrame: Parsed data with initial type inference

        Raises:
            RuntimeError:                Unsupported file format or read failure
            ParserError:                 Malformed CSV/Excel content
        """
        try:
            if file_path.suffix.lower() in csv_ext:
                if chunksize:
                    chunks = list(pd.read_csv(file_path, chunksize=chunksize, **csv_kwargs))
                    return pd.concat(chunks, ignore_index=True)
                return pd.read_csv(file_path, **csv_kwargs)
            
            if file_path.suffix.lower() in excel_ext:
                return pd.read_excel(file_path, **excel_kwargs)
            
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        except Exception as e:
            raise RuntimeError(
                f"Error reading {file_path.name}: {str(e)}"
            ) from e

    @staticmethod
    def _post_merge_cleanup(
        df: pd.DataFrame, 
        date_col: str | None, 
        id_cols: List[str] | None
    ) -> pd.DataFrame:
        """
        Performs post-merge dataset optimization and validation

        Parameters:
            df (pd.DataFrame):           Raw combined dataset
            date_col (str | None):       datetime column name
            id_cols (List[str] | None):  categorical columns

        Returns:
            pd.DataFrame: Optimized dataset with:
                - Deduplicated records
                - Valid datetime conversion
                - Memory-optimized dtypes

        Raises:
            ValueError:                  Invalid datetime values in date_col
        """
        # Deduplication
        df = df.drop_duplicates().reset_index(drop=True)
        
        # Vectorized date parsing
        if date_col:
            df[date_col] = pd.to_datetime(
                df[date_col],
                errors='coerce'
            )
            if df[date_col].isna().all():
                raise ValueError(f"All values in date column '{date_col}' are invalid")
            df = df.sort_values(date_col)
        
        # Memory optimization
        return ProductionDataAnalyzer._optimize_dtypes(df, date_col, id_cols)

    @staticmethod
    def _optimize_dtypes(
        df: pd.DataFrame, 
        date_col: str | None, 
        id_cols: List[str] | None
    ) -> pd.DataFrame:
        """
        Optimizes DataFrame memory usage through type downcasting

        Parameters:
            df (pd.DataFrame):           Input dataset
            date_col (str | None):       Already converted datetime column
            id_cols (List[str] | None):  Columns to convert to categorical

        Returns:
            pd.DataFrame: Memory-optimized dataset with:
                - Categorical types for ID columns
                - Downcast numeric types
                - String types for high-cardinality text
                - Boolean types for binary values

        Raises:
            TypeError:                   Invalid type conversion attempts
        """
        for col in df.columns:
            if col == date_col:
                continue
            
            if id_cols is not None and col in id_cols:
                df[col] = df[col].astype('category')
                continue

            if pd.api.types.is_string_dtype(df[col]):
                # Attempt numeric conversion
                numeric = pd.to_numeric(df[col], errors='coerce')
                if numeric.notna().mean() > 0.9:
                    df[col] = numeric.pipe(pd.to_numeric, downcast='unsigned')
                else:
                    unique_ratio = df[col].nunique() / len(df)
                    df[col] = df[col].astype('category' if unique_ratio < 0.5 else 'string')
            
            elif pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], downcast='integer' if 'int' in str(df[col].dtype) else 'float')
        
        return df

    @staticmethod
    def filter_by_id(
        production_data_df: pd.DataFrame,
        id_data_df: pd.DataFrame,
        id_col: str
    ) -> pd.DataFrame:
        """
        Filter a production DataFrame based on IDs present in a reference DataFrame

        Parameters:
            production_data_df (pd.DataFrame):  DataFrame containing production data to filter
            id_data_df (pd.DataFrame):          Reference DataFrame containing valid IDs
            id_col (str):                       Column name containing IDs in both DataFrames

        Returns:
            pd.DataFrame: Filtered production data containing only rows with IDs present in reference data

        Raises:
            KeyError:   If the specified ID column is missing from either DataFrame
            TypeError:  If the ID columns have incompatible data types
            ValueError: If input DataFrames are empty or no matching IDs found

        Example:
            >>> production_df = pd.DataFrame({
            ...     'ID': [101, 102, 103],
            ...     'value': [25, 30, 35]
            ... })
            >>> id_df = pd.DataFrame({'ID': [102, 103]})
            >>> filtered = ProductionDataAnalyzer.filter_by_id(production_df, id_df, 'ID')
            >>> print(filtered)
              ID  value
            1  102     30
            2  103     35

        Notes:
            - Performs case-sensitive comparison for string IDs
            - Maintains original row order from production data
            - Returns a copy of the filtered data to prevent SettingWithCopy warnings
            - Converts ID columns to string type for cross-type matching
            - ID columns are converted to strings during matching
            - Numeric IDs will be stringified (e.g., 00123 → '123')
            - For exact matching of zero-padded IDs, ensure ID columns  are stored as strings in your source data
        """
        # Validate input DataFrames
        if production_data_df.empty or id_data_df.empty:
            raise ValueError("Input DataFrames cannot be empty")

        # Verify ID column existence
        if id_col not in production_data_df.columns:
            raise KeyError(f"ID column '{id_col}' not found in production data")
        if id_col not in id_data_df.columns:
            raise KeyError(f"ID column '{id_col}' not found in reference data")

        # Convert ID columns to string for type safety
        try:
            production_ids = production_data_df[id_col].astype(str)
            reference_ids = id_data_df[id_col].astype(str).unique()
        except TypeError:
            raise TypeError("ID columns could not be converted to string type")

        # Create filter mask
        filter_mask = production_ids.isin(reference_ids)
        
        # Check for matches
        if not filter_mask.any():
            raise ValueError("No matching IDs found between datasets")

        # Return filtered copy of data
        filtered_df = production_data_df.loc[filter_mask].copy()
        
        # Reset index while preserving original order
        return filtered_df.reset_index(drop=True)

    def save_to_csv(
        self,
        df: pd.DataFrame,
        filename: str = 'output.csv',
        sep: str = ';',
        decimal: str = ',',
        force_download: bool = False
    ) -> None:
        """
        Save DataFrame to CSV with environment-adapted behavior

        Parameters:
            df (pd.DataFrame):      DataFrame to save
            filename (str):         Output filename/path
            sep (str):              Column separator
            decimal (str):          Decimal separator
            force_download (bool):  Bypass confirmation prompt (local only)
                                    Default: False

        Environment-specific Behavior:
            Colab:
                - Triggers browser download automatically
                - Saves temporary file in Colab runtime
            Local:
                - Saves to filesystem
                - Shows rich confirmation prompt unless force_download=True
                - Allows filename modification via prompt

        Raises:
            ValueError: For invalid filename/DataFrame
            PermissionError: For write permission issues
            RuntimeError: For download failures in Colab

        Example:
            >>> analyzer.save_to_csv(df, 'production_data.csv')
                Save file to production_data.csv? [y/N]: y
                Saved to /projects/data/production_data.csv (25.6KB)
        """
        # Input validation
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a DataFrame")
        if len(df) < 1:
            raise ValueError("DataFrame must contain at least one row")
        
        if not isinstance(filename, str) or not filename.endswith('.csv'):
            raise ValueError("Filename must be string ending with .csv")

        if len(sep) != 1 or len(decimal) != 1:
            raise ValueError("Separators must be single-character strings")

        # Common saving logic
        def save_file(path: str) -> float:
            from pathlib import Path
            output_path = Path(filename)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(
                output_path,
                index=False,
                sep=sep,
                decimal=decimal,
                encoding='utf-8',
                date_format='%Y-%m-%d %H:%M:%S'
            )
            return os.path.getsize(path) / 1024

        try:
            if self.IN_COLAB:
                # Colab: Save and trigger download
                file_size = save_file(filename)
                from google.colab import files
                files.download(filename)
                self._print(
                    f"Saved and downloaded [bold green]{filename}[/] "
                    f"({len(df):,} rows, {file_size:.1f}KB)",
                    style="green"
                )
                
            else:
                # Local: Interactive handling
                from rich.prompt import Confirm, Prompt
                
                if force_download:
                    file_size = save_file(filename)
                    self._print(
                        f"Saved to [bold green]{os.path.abspath(filename)}[/] "
                        f"({len(df):,} rows, {file_size:.1f}KB)",
                        style="green"
                    )
                else:
                    if Confirm.ask(
                        f"Save to [bold]{filename}[/]?",
                        default=True
                    ):
                        file_size = save_file(filename)
                        self._print(
                            f"Saved to [bold green]{os.path.abspath(filename)}[/] "
                            f"({len(df):,} rows, {file_size:.1f}KB)",
                            style="green"
                        )
                    else:
                        new_name = Prompt.ask(
                            "Enter new filename",
                            default=filename
                        )
                        if not new_name.endswith('.csv'):
                            new_name += '.csv'
                            
                        file_size = save_file(new_name)
                        self._print(
                            f"Saved to [bold green]{os.path.abspath(new_name)}[/] "
                            f"({len(df):,} rows, {file_size:.1f}KB)",
                            style="green"
                        )

        except PermissionError as pe:
            raise PermissionError(
                f"Permission denied for [bold]{filename}[/]: "
                f"{str(pe)}"
            ) from pe
            
        except Exception as e:
            raise RuntimeError(
                f"[red]Failed to save CSV:[/] {str(e)}"
            ) from e
      
    def aggregate_data(self, period: str = 'day') -> None:
        """
        Aggregate data to specified time periods

        Parameters:
            period (str):   Temporal aggregation interval. Valid options:
                - 'day':    Daily aggregation (default)
                - 'week':   Weekly aggregation starting Mondays
                - 'month':  Monthly aggregation from month start
                - 'year':   Yearly aggregation from year start

        Returns:
            None: Updates instance attributes in-place:
                - daily_data (pd.DataFrame): Aggregated dataset
                - selected_params (list): Numeric parameters used in aggregation
                - agg_freq (str): Pandas frequency string used for resampling
                - period (str): Human-readable aggregation period name

        Raises:
            ValueError: If any of these conditions occur:
                - Invalid period specification
                - Missing datetime column initialization
                - No numeric parameters available for aggregation
                - Datetime column contains invalid/out-of-order timestamps

        Example:
            >>> analyzer = ProductionDataAnalyzer(df, date_col='timestamp')
            >>> analyzer.aggregate_data(period='week')
            Aggregated to 52 week intervals
            >>> analyzer.daily_data.head()
              timestamp  temperature  pressure  vibration
            0 2023-01-02      72.4      105.2      4.8
            1 2023-01-09      71.9      106.1      5.2

        Notes:
            - Aggregation uses mean calculation for numeric parameters
            - Automatically excludes temporal metadata columns (year/month/week/day)
            - Validates datetime column integrity before resampling
            - Maintains original timezone information if present
        """
        # Validate period input
        period_map = {
            'day': 'D',
            'week': 'W-MON',
            'month': 'MS',
            'year': 'YS'
        }
        
        # Convert to Dask DataFrame for parallel processing
        ddf = dd.from_pandas(self.production.set_index(self.date_col), npartitions=4)

        # Perform resampling and aggregatrion in parallel
        aggregated = ddf.resample(period_map[period]).mean().compute()

        # Convert back to pandas and format
        self.aggregate_data = aggregated.round(4)
        self._print(f"Aggregated data shape: {self.aggregate_data.shape}")

    def save_aggregated_data(self, filename: str = 'aggregated_data.csv') -> None:
        """
        Save aggregated data to CSV file and initiate file download (Google Colab only)

        Preserves European-style numeric formatting using:
            - ';' as separator
            - '.' as decimal separator
            - UTF-8 encoding

        Parameters:
            filename (str): Output filename
                Default: 'aggregated_data.csv'

        Returns:
            None: Downloads the file to the local machine

        Raises:
            ValueError:       If no aggregated data is available or filename is invalud
            PermissionError:  If write permissions are insufficient for target location
            RuntimeError:     If called outside Google Colab environment

        Example:
            >>> analyzer = ProductionDataAnalyzer(df, date_col='timestamp')
            >>> analyzer.aggregate_data(period='week')
            >>> analyzer.save_aggregated_data('weekly_stats.csv')
            Data saved to weekly_stats.csv (456 rows, 12 parameters)
            Downloading weekly_stats.csv...

        Notes:
            - Requires prior execution of aggregate_data() method
            - In local environments, file will be saved but not auto-downloaded
            - Preserves datetime formatting from aggregated_data index
            - Maintains categorical data encoding from original dataset
        """
        # Validate internal state
        if self.daily_data.empty:
            raise ValueError(
                "No aggregated data available. Run aggregate_data() first."
        )
        
        # Validate filename
        if not isinstance(filename, str) or not filename.endswith('.csv'):
            raise ValueError("Filename must be string with .csv extension")

        try:
            # Save with European CSV formatting
            self.daily_data.to_csv(
                filename,
                index=False,
                sep=';',
                decimal=',',
                encoding='utf-8',
                date_format='%Y-%m-%d %H:%M:%S'
            )
        except PermissionError as pe:
            raise PermissionError(
                f"Write permission denied for {filename}"
                "Check directory permissions or try different location"
            ) from pe
        except Exception as e:
            raise RuntimeError(
                f"Failed to save {filename}: {str(e)}"
            ) from e

        except ImportError:
            print(f"File saved locally at {os.path.abspath(filename)}")
            return

        # Provide detailed output summary
        row_count = len(self.daily_data)
        param_count = len(self.selected_params)
        file_size_kb = os.path.getsize(filename) / 1024
        print(
            f"Data saved to {filename}"
            f"({row_count} rows, {param_count} parameters, {file_size_kb}KB)"
        )

        # Handle Colab-specific download
        try:
            from google.colab import files
            files.download(filename)
        except ImportError:
            raise RuntimeError(
                "This function is only available in Google Colab"
                f"File saved locally to {os.path.abspath(filename)}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download {filename}: {str(e)}"
                f"File remains available at {os.path.abspath(filename)}"
            ) from e

    def set_parameter_limits(self, source: Union[str, Dict, pd.DataFrame] = None) -> None:
        """
        Configure parameter(s) limits based on source
        (Google Sheets URL, dictionary or DataFrame loaded by ProductionDataAnalyzer.upload_files())

        Parameters:
            source: Input source containing limits, can be:
                - Google Sheets URL (str)
                - Uploaded file (CSV, Excel)
                - Dictionary {param: (LSL, USL)}

        Raises:
            ValueError: For invalid limits or format errors
            RuntimeError: For Google Sheets authentication failures

        Example Dictionary:
            {'fitting_force': (1000, 2200), 'fitting_height': (20, 20.5)}

        Example CSV Format:
            parameter,LSL,USL
            fitting_force,1000,2200
            fitting_height,20,20.5

        Returns:
            None: Updates instance attributes in-place:
                - param_limits (dict): Dictionary with parameter(s) limits
        """
        limits = {}

        try:
            if isinstance(source, dict):
                limits = self._process_dict_source(source)
            elif isinstance(source, str) and 'docs.google.com' in source:
                limits = self._process_gsheet_source(source)
            elif isinstance(source, pd.DataFrame):
                limits = self._process_dataframe_source(source)
            else:
                raise ValueError("Unsupported source type")
        except Exception as e:
            raise RuntimeError(f"Failed to process limits source: {str(e)}") from e

        self._validate_and_store_limits(limits)

    def _process_dict_source(self, source: Dict) -> Dict:
        """
        Process dictionary input with validation and conversion

        Parameters:
            source (Dict): Input dictionary in format {parameter: (LSL, USL)}

        Returns:
            Dict: Processed limits as {parameter: (LSL, USL)} with float values

        Raises:
            TypeError:      For invalid data types
            ValueError:     For data integrity issues
            RuntimeError:   For unexpected processing failures

        Example valid input:
            {'fitting_force': (1000, 2200), 'fitting_height': (20, 20.5)}

        Example invalid input:
            {123: (20, 30)}                 → TypeError
            {'': (10, 20)}                  → ValueError
            'fitting_force': [30]           → ValueError
            'fitting_force': ('low', 50)    → TypeError
            'fitting_height': (120, 80)     → ValueError
        """
        processed = {}

        try:
            if not isinstance(source, dict):
                raise TypeError("Input must be a dictionary")

            for param, lim in source.items():
                # Validate parameter name
                if not isinstance(param, str):
                    raise TypeError(
                        f"Parameter name '{param}' must be string (got {type(param)})"
                    )
                
                clean_param = param.strip()
                if not clean_param:
                    raise ValueError("Empty parameter name found")

                # Validate limits structure
                if not isinstance(lim, (list, tuple)) or len(lim) != 2:
                    raise ValueError(
                        f"Invalid limits format for {param}. "
                        f"Expected 2-element sequence, got {type(lim)} with {len(lim)} elements"
                    )

                # Convert to floats
                try:
                    lower = float(lim[0])
                    upper = float(lim[1])
                except (TypeError, ValueError) as e:
                    raise TypeError(
                        f"Non-numeric limits for {param}: {lim[0]!r}, {lim[1]!r}"
                    ) from e

                # Validate limit relationship
                if lower >= upper:
                    raise ValueError(
                        f"Invalid limits for {param}: "
                        f"Lower ({lower}) ≥ Upper ({upper})"
                    )

                processed[clean_param] = (lower, upper)

            return processed

        except Exception as e:
            if isinstance(e, (TypeError, ValueError)):
                raise
            raise RuntimeError(f"Dictionary processing failed: {str(e)}") from e

    def _process_gsheet_source(self, url: str) -> Dict:
        """
        Validate and process Google Sheets input

        Parameters:
            url (str): Google Sheets URL containing parameter(s) limits
                - Standard edit URL: https://docs.google.com/spreadsheets/d/<ID>/edit...
                - Published CSV URL: https://docs.google.com/spreadsheets/d/<ID>/export?format=csv

        Returns:
            Dict: Processed limits as {parameter: (LSL, USL)}

        Raises:
            ValueError: For invalid URL format, empty worksheets or inaccessible sheets
            RuntimeError: For authentication failures, API errors or CSV export failures
            TypeError: For non-string URL input
            Propagates: Data validation errors from DataFrame processing

        Example valid input:
            'https://docs.google.com/spreadsheets/d/abc123/edit#gid=0'

        Example invalid cases:
            Invalid URL format: 'https://example.com'   → ValueError
            Blocked auth + unpublished sheet            → RuntimeError
            Non-string input: 12345                     → TypeError
            Empty published sheet                       → ValueError
        """
        try:
            # Validate input type
            if not isinstance(url, str):
                raise TypeError(f"URL must be string, got '{type(url)}'")

            # Try authenticated API access first
            try:
                return self._process_gsheet_via_api(url)
            except Exception as api_error:
                # Fall back to CSV export if API access failed
                try:
                    return self._process_gsheet_via_csv(url)
                except Exception as csv_error:
                    # Combine error information
                    raise RuntimeError(
                        "Failed to access Google Sheet through both methods:\n"
                        f"API Error: {str(api_error)}\n"
                        f"CSV Error: {str(csv_error)}\n"
                        "For CSV fallback, ensure:\n"
                        "- File → Share → Publish to web\n"
                        "- Select 'Comma-separated values (.csv)'"
                    ) from csv_error

        except Exception as e:
            if isinstance(e, (TypeError, ValueError, RuntimeError)):
                raise
            raise RuntimeError(f"Unexpected error processing Google Sheet: {str(e)}") from e

    def _process_gsheet_via_api(self, url: str) -> Dict:
        """
        Process Google Sheet input via authenticated API access

        Paramters:
            url (str): Google Sheets URL with edit access

        Returns:
            Dict: Processed limits from first worksheet

        Raises:
            ValueError: For invalid URL, empty sheets or access issues
            RuntimeError: For authentication/API failures
            gspread.exceptions.APIError: For Google API issues

        Example failure cases:
            Corporate authentication/firewall blocking  → RuntimeError
            Sheet not shared with service account       → ValueError
            Worksheet contains no data                  → ValueError
        """
        # URL validation
        if 'docs.google.com' not in url or '/spreadsheets/' not in url:
            raise ValueError(
                f"Invalif Google Sheets URLL {url[:50]}...\n"
                "Required format: 'https://docs.google.com/spreadsheets/d/[ID]/edit"
            )

        try:
            # Authentication flow
            auth.authenticate_user()
            creds, _ = default()
            gc = gspread.authorize(creds)

            # Sheet access
            try:
                sheet = gc.open_by_url(url)
            except gspread.SpreadsheetNotFound:
                available_sheets = gc.openall()
                raise ValueError("Sheet not found") from None

            # Worksheet handling
            try:
                worksheet = sheet.get_worksheet(0)
            except IndexError:
                raise ValueError("Document contains no worksheets") from None

            # Data validation
            records = worksheet.get_all_records()
            if not records:
                raise ValueError(
                    f"Worksheet '{worksheet.title}' empty (headers: {worksheet.row_values(1)})"
                )

            return self._process_dataframe_source(pd.DataFrame(records))
        
        except gspread.exceptions.AuthenticationError as e:
            raise RuntimeError(
                "Google Sheets authentication blocked. Possible reasons:\n"
                "- Corporate firewall/settings restrictions\n"
                "- Missing required permissions\n"
                "- Invalid credentials\n"
                "Try CSV fallback method instead"
            ) from e

        except (gspread.exceptions.APIError, AttributeError) as e:
            if "unauthorized" in str(e).lower():
                raise RuntimeError("Authentication failed") from e

    def _process_gsheet_via_csv(self, url: str) -> Dict:
        """
        Process Google Sheet input via public CSV export

        Parameters:
            url (str): Google Sheets URL

        Returns:
            Dict: Processed limits from specified worksheet

        Raises:
            ValueError:     For unpublished sheets or invalid gid
            RuntimeError:   For CSV parsing failures
            KeyError:       If required columns missing

        Example failure cases:
            Sheet not published to web  → ValueError
            Invalid worksheet gid       → ValueError
            Modified CSV structure      → KeyError
        """
        from urllib.parse import urlparse, parse_qs

        try:
            # Extract sheet ID
            parsed = urlparse(url)
            if 'spreadsheets' not in parsed.path:
                raise ValueError("Not a Google Sheets URL")
                
            path_parts = parsed.path.split('/')
            sheet_id = path_parts[path_parts.index('d') + 1] if 'd' in path_parts else None
            
            if not sheet_id or len(sheet_id) < 5:
                raise ValueError(f"Invalid sheet ID in URL: {url[:50]}...")
            
            # Extract worksheet ID (gid)
            gid = '0'
            if 'gid=' in parsed.fragment:
                gid = parse_qs(parsed.fragment)['gid'][0]
            elif 'gid=' in parsed.query:
                gid = parse_qs(parsed.query)['gid'][0]

            # Build CSV export URL
            csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
            
            # Read and validate CSV
            try:
                df = pd.read_csv(csv_url)
            except pd.errors.ParserError:
                raise ValueError(
                    "CSV export failed. Verify sheet is published:\n"
                    "- File → Share → Publish to web\n"
                    "- Select 'Comma-separated values (.csv)'"
                ) from None

            if df.empty:
                raise ValueError(
                    "Published sheet contains no data. Check:\n"
                    "- Worksheet has data below header\n"
                    "- Published range includes data\n"
                    "- Refresh publication settings"
                )

            return self._process_dataframe_source(df)

        except (IndexError, KeyError) as e:
            raise ValueError(
                f"URL parsing failed: {url[:50]}...\n"
                "Required format: 'https://docs.google.com/spreadsheets/d/[ID]/edit#gid=[NUM]'"
            ) from e

    def _process_dataframe_source(self, df: pd.DataFrame) -> Dict:
        """
        Validate and process DataFrame input

        Returns:
            Dict: {parameter: (LSL, USL)}

        Raises:
            ValueError: For data integrity issues
            TypeError: For data type mismatches
            RuntimeError: Foe unexpected processing failures
        """
        try:
            # Check mandatory columns
            required_columns = {'parameter', 'LSL', 'USL'}
            missing_columns = required_columns - set(df.columns)
            if missing_columns:
                raise ValueError(
                    f"Missing required columns: {missing_columns}\n"
                    f"Existing columns: {list(df.columns)}"
                )

            # Remove potential whitespace in parameter names
            df['parameter'] = df['parameter'].str.strip()

            # Check for empty parameters
            empty_params = df[df['parameter'].isnull() | (df['parameter'] == '')]
            if not empty_params.empty:
                raise ValueError(
                    f"Empty parameter names found at rows: {empty_params.index.tolist()}"
                )
            
            # Validate numeric limits
            for col in ['USL', 'LSL']:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    invalid_rows = df[pd.to_numeric(df[col], errors='coerce').isnull()]
                    raise TypeError(
                        f"Non-numeric values in {col} column:\n"
                        f"{invalid_rows[[col]].to_string()}"
                    )
            
            # Check LSL < USL consistency
            invalid_limits = df[df['LSL'] >= df['USL']]
            if not invalid_limits.empty:
                error_list = "\n".join(
                    f"- {row['parameter']}: LSL={row['LSL']} ≥ USL={row['USL']}"
                    for _, row in invalid_limits.iterrows()
                )
                raise ValueError(
                    f"Lower limit exceeds upper limit for parameters:{error_list}"
                )

            # Check for duplicate parameters
            duplicates = df[df.duplicated('parameter', keep=False)]
            if not duplicates.empty:
                dupe_list = '\n'.join(
                    f"- {param} ({count} entries)"
                    for param, count in duplicates['parameter'].value_counts().items()
                )
                raise ValueError(
                    f"Duplicate parameter entries found: \n{dupe_list}"
                )

            # Convert to formated dictionary
            return df.set_index('parameter')[['LSL', 'USL']].apply(tuple, axis=1).to_dict()

        except KeyError as ke:
            raise ke
        except pd.errors.ParserError as pe:
            raise pe
        except (ValueError, TypeError) as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"Unexpected processing error: {str(e)}") from e

    def _validate_and_store_limits(self, new_limits: Dict) -> None:
        """
        Validate and merge parameter(s) limits with existing configuration
        All the validations are redundant (already done in processing) but oh well, just in case :)

        Parameters:
            new_limits (Dict): {parameter: (LSL, USL),...} mapping to add/update

        Raises:
            TypeError: For invalid input types
            ValueError: For data integrity violations
            RuntimeError: For unexpected validation failures

        Example valid input:
            {'fitting_force': (1000, 2200), 'fitting_height': (20, 20.5)}

        Example invalid input:
            Non-dictionary input            → TypeError
            Parameter not in dataset        → ValueError
            Non-numeric parameter column    → ValueError
            Lower ≥ upper limit             → ValueError
        """
        try:
            # Input type validation
            if not isinstance(new_limits, dict):
                raise TypeError(
                    f"Limits must be dictionary, got '{type(new_limits)}'"
                )
            if not new_limits:
                raise ValueError("Cannot store empty limits dictionary")

            valid_params = set(self.production.columns)
            numeric_params = set(
                self.production.select_dtypes(include=np.number).columns
            )
            existing_params = set(self.param_limits.keys())

            errors = []
            warnings = []
            clean_limits = {}

            # Pre-validation for parameters existence and types
            for param, limits in new_limits.items():
                # Parameter name validation
                if not isinstance(param, str):
                    errors.append(
                        f"Invalid parameter name type '{type(param)}' "
                        f"for {param} - must be string"
                    )
                    continue
                
                # Parameter existence check
                clean_param = param.strip()
                if not clean_param:
                    errors.append("Empty parameter name found")
                    continue
                if clean_param not in valid_params:
                    errors.append(f"Parameter '{clean_param}' not in dataset")
                    continue

                # Limits structure validation
                if not isinstance(limits, (tuple, list)) or len(limits) != 2:
                    errors.append(
                        f"Invalid limits format for {clean_param}: "
                        f"Expected 2-element sequence, got {type(limits)} "
                        f"with {len(limits)} elements"
                    )
                    continue

                # Numeric validation
                lower, upper = limits
                for val, pos in [(lower, "LSL"), (upper, "USL")]:
                    if not isinstance(val, (int, float)):
                        errors.append(
                            f"{pos} limit for {clean_param} must be numeric, "
                            f"got '{type(val)}'"
                        )
                    if pd.isna(val):
                        errors.append(
                            f"{pos} limit for {clean_param} cannot be NaN"
                        )

                # Numeric parameter check
                if clean_param not in numeric_params:
                    warnings.append(
                        f"Parameter '{clean_param}' exists but is non-numeric "
                        f"(dtype: {self.production[clean_param].dtype})"
                    )

                # Limit relationship validation
                if not errors[-1:]:
                    if lower >= upper:
                        errors.append(
                            f"Invalid limits for {clean_param}: "
                            f"LSL={lower} ≥ USL={upper}"
                        )
                
                if not errors[-1:]:
                    clean_limits[clean_param] = (float(lower), float(upper))

            # Raise collected errors
            if errors:
                error_msg = "Validation failed:" + "\n - ".join(errors)
                raise ValueError(error_msg)

            # Show warnings
            if warnings:
                print("Validation warnings:" + "\n - ".join(warnings))

            # Track changes
            new_params = set(clean_limits.keys()) - existing_params
            updated_params = set(clean_limits.keys()) & existing_params

            # Update stored limits
            self.param_limits.update(clean_limits)

            # Generate summary
            summary = [
                "Successfully stored parameter(s) limits:",
                f"- New parameters: {len(new_params)}",
                f"- Updated parameters: {len(updated_params)}",
                f"Total configured parameters: {len(self.param_limits)}",
                f"Dataset parameters available: {len(numeric_params)}"
            ]
            print("\n".join(summary))

        except Exception as e:
            if isinstance(e, (TypeError, ValueError)):
                raise
            raise RuntimeError(
                f"Limit storage failed: {str(e)}"
            ) from e

    def generate_eda_report(
        self,
        sample_size: int = None,
        console_width: int = 120,
        max_correlations: int = 10,
        output_path: Optional[str] = None
    ) -> Union[Dict, str]:
        """
        Generate environment-appropriate EDA report with technical insights

        Parameters:
            sample_size (int):         Maximum samples for visualizations
                                        (terminal mode enforces 5000 max)
            console_width (int):        Width of terminal output (characters)
                                        Default: 120
            max_correlations (int):     Maximum correlation pairs to show
                                        Default: 10
            output_path (str):          File path to save HTML report (Colab only)
                                        Default: None

        Environment-specific Behavior:
            Colab:
                - Returns Dict with interactive Plotly figures
                - Optionally saves HTML report
            Local:
                - Outputs rich terminal report
                - Returns formatted string if rich disabled

        Returns:
            Dict: Colab - Full report dictionary with figures
            str:  Local - Formatted text report when rich disabled

        Raises:
            RuntimeError: If report generation fails in current environment
        """
        report_data = {}
        
        try:
            # Core report components
            report_data['summary_stats'] = self._get_summary_stats()
            report_data['missing_data'] = self._analyze_missing_data()
            report_data['distribution_plots'] = self.plot_feature_distributions(sample_size)
            report_data['correlations'] = self.analyze_correlations()
            
            if self.date_col:
                report_data['temporal_trends'] = self.plot_interactive_timeline()
            else:
                report_data['temporal_trends'] = None
            
            # Statistical insights
            report_data['statistical_insights'] = self._generate_statistical_insights(
                report_data['summary_stats']
            )
        except Exception as e:
            raise RuntimeError(f"EDA report generation failed: {str(e)}") from e

        if self.IN_COLAB:
            # Colab-specific output handling
            if output_path:
                self._save_html_report(report_data, output_path)
            return report_data
        else:
            return self._render_local_report(report_data, console_width, max_correlations)

    def _render_local_report(
        self,
        report_data: Dict,
        console_width: int,
        max_correlations: int
    ) -> Optional[str]:
        """Render terminal-optimized EDA report."""
        if self.enable_rich:
            self._display_rich_report(report_data, console_width, max_correlations)
            return None
        else:
            return self._format_text_report(report_data, max_correlations)

    def _display_rich_report(
        self,
        report_data: Dict,
        width: int,
        max_correlations: int
    ) -> None:
        """Rich terminal report presentation."""
        from rich.panel import Panel
        from rich.columns import Columns

        # Summary Stats
        summary_table = Table(title="Numerical Summary", width=width-4)
        summary_table.add_column("Feature", style="cyan")
        for col in report_data['summary_stats'].columns:
            summary_table.add_column(col, style="magenta")
        
        for idx, row in report_data['summary_stats'].iterrows():
            summary_table.add_row(idx, *[str(self._smart_round(v)) for v in row])

        self.console.print(Panel(summary_table, title="[bold]1. Summary Statistics"))

        # Missing Data
        missing_table = Table(title="Missing Values", width=width-4)
        missing_table.add_column("Feature", style="cyan")
        missing_table.add_column("Missing", style="red")
        missing_table.add_column("Percentage", style="yellow")
        
        for idx, row in report_data['missing_data'].iterrows():
            missing_table.add_row(
                idx,
                str(row['missing_count']),
                f"{row['missing_pct']:.1%}"
            )
        
        self.console.print(Panel(missing_table, title="[bold]2. Missing Data Analysis"))

        # Statistical Insights
        insights_panel = Panel(
            Markdown(report_data['statistical_insights']),
            title="[bold]3. Technical Insights",
            width=width
        )
        self.console.print(insights_panel)

        # Correlation Analysis
        corr_matrix = self._format_correlation_matrix(
            report_data['correlations']['pearson'],
            max_correlations
        )
        self.console.print(Panel(corr_matrix, title="[bold]4. Top Correlations"))

        # Temporal Trends
        if 'temporal_trends' in report_data:
            self.console.print(Panel.fit(
                "[bold green]Temporal trends available - call plot_interactive_timeline()",
                title="[bold]5. Temporal Analysis"
            ))

    def _format_correlation_matrix(
        self,
        corr_matrix: pd.DataFrame,
        max_pairs: int
    ) -> Table:
        """Create rich table of top correlations."""
        corr_table = Table(title=f"Top {max_pairs} Correlations")
        corr_table.add_column("Pair", style="cyan")
        corr_table.add_column("Pearson", style="magenta")
        corr_table.add_column("Type", style="yellow")

        pairs = corr_matrix.unstack().sort_values(key=abs, ascending=False)
        count = 0
        
        for (f1, f2), value in pairs.items():
            if f1 == f2 or count >= max_pairs:
                continue
                
            corr_type = "🡅 Positive" if value > 0 else "🡇 Negative"
            style = "green" if abs(value) > 0.7 else "yellow" if abs(value) > 0.5 else "dim"
            
            corr_table.add_row(
                f"{f1} ↔ {f2}",
                f"{value:.2f}",
                corr_type,
                style=style
            )
            count += 1

        return corr_table

    def _format_text_report(
        self,
        report_data: Dict,
        max_correlations: int
    ) -> str:
        """Generate plain text EDA report."""
        report = []
        
        # Summary Stats
        report.append("=== Numerical Summary ===")
        report.append(report_data['summary_stats'].to_string())
        
        # Missing Data
        report.append("\n=== Missing Values ===")
        report.append(report_data['missing_data'].to_string())
        
        # Statistical Insights
        report.append("\n=== Technical Insights ===")
        report.append(report_data['statistical_insights'])
        
        # Correlations
        report.append("\n=== Top Correlations ===")
        pairs = report_data['correlations']['pearson'].unstack()
        pairs = pairs.sort_values(key=abs, ascending=False)
        
        count = 0
        for (f1, f2), value in pairs.items():
            if f1 != f2 and count < max_correlations:
                report.append(f"{f1} vs {f2}: {value:.2f}")
                count += 1
        
        return "\n".join(report)

    def _save_html_report(self, report_data: Dict, output_path: str) -> None:
        """Save interactive HTML report (Colab only)."""
        from plotly.io import to_html
        
        html_content = []
        for section in ['distribution_plots', 'correlations', 'temporal_trends']:
            if fig := report_data.get(section):
                html_content.append(to_html(fig))
        
        with open(output_path, 'w') as f:
            f.write("<html><body>")
            f.write("<h1>Production Data EDA Report</h1>")
            f.write("\n".join(html_content))
            f.write("</body></html>")
        
        print(f"Saved HTML report to {output_path}")

    def _get_summary_stats(self) -> pd.DataFrame:
        """
        Generate statistical summary for numeric parameters

        Returns:
            pd.DataFrame: Statistical summary with columns:
                - count: Number of non-null values
                - mean: Average value
                - std: Standard deviation
                - min: Minimum value
                - 1%: 1st percentile
                - 25%: 25th percentile (Q1)
                - 50%: Median
                - 75%: 75th percentile (Q3)
                - 99%: 99th percentile
                - max: Maximum value

        Raises:
            ValueError: If no numeric columns exist in dataset

        Note:
            Excludes non-numeric columns from calculations
        """
        numeric_cols = self.production.select_dtypes(include=np.number).columns
        if not numeric_cols.empty:
            return self.production[numeric_cols].describe(percentiles=[.01, .25, .5, .75, .99]).T.round(4)
        raise ValueError("No numeric columns found for statistical summary")

    def _analyze_missing_data(self):
        """
        Analyze missing value patterns across dataset columns

        Returns:
            pd.DataFrame: Missing data analysis with columns:
                - missing_count: Number of missing values
                - missing_pct: Percentage of missing values
                - data_type: Column data type
                - unique_count: Number of unique values

        Raises:
            RuntimeError: If dataset contains no columns
        """
        if self.production.empty:
            raise RuntimeError("Cannot analyze missing data - dataset is empty")
            
        missing = self.production.isna().sum().to_frame('missing_count')
        missing['missing_pct'] = missing['missing_count'] / len(self.production)
        missing['data_type'] = self.production.dtypes
        missing['unique_count'] = self.production.nunique()
        return missing.sort_values('missing_pct', ascending=False)

    def plot_feature_distributions(
        self,
        sample_size: int = None,
        terminal_width: int = 80,
        terminal_height: int = 20,
        hist_bins: int = 20
    ) -> Optional[go.Figure]:
        """
        Create interactive distribution plots with statistical summary table

        Parameters:
            sample_size (int):         Maximum samples for visualization
                                        (terminal mode enforces 5000 max)
            terminal_width (int):      Width of terminal plots (characters)
                                        Default: 80
            terminal_height (int):     Height of terminal plots (lines)
                                        Default: 20
            hist_bins (int):           Number of histogram bins
                                        Default: 20

        Environment-specific Behavior:
            Colab:
                - Returns interactive Plotly Figure
                - Uses full dataset unless sample_size specified
            Local:
                - Outputs terminal plots directly
                - Enforces sample_size <= 5000 for performance
                - Returns None

        Returns:
            go.Figure | None: Plotly figure in Colab, None in local mode

        Raises:
            ValueError: If no numeric columns found
            RuntimeError: If visualization fails in current environment
        """
        numeric_cols = self.production.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric columns available for distribution analysis")

        # Prepare data sample
        df = self.production if sample_size is None else \
            self.production.sample(min(sample_size, len(self.production)))

        if self.IN_COLAB:
            # Plotly implementation for Colab
            fig = make_subplots(
                rows=len(numeric_cols), cols=2,
                specs=[[{"type": "histogram"}, {"type": "box"}]] * len(numeric_cols),
                subplot_titles=[f"{col} | Histogram" for col in numeric_cols] + 
                            [f"{col} | Box Plot" for col in numeric_cols]
            )

            for idx, col in enumerate(numeric_cols, 1):
                fig.add_trace(go.Histogram(x=df[col], name=col), row=idx, col=1)
                fig.add_trace(go.Box(x=df[col], name=col), row=idx, col=2)

            fig.update_layout(
                height=300*len(numeric_cols),
                showlegend=False,
                title_text="Feature Distributions"
            )
            return fig

        else:
            # Terminal visualization for local environment
            max_terminal_samples = min(5000, len(df))
            if len(df) > max_terminal_samples:
                df = df.sample(max_terminal_samples)
                self._print(f"[yellow]Sampled {max_terminal_samples} records for terminal display[/]")

            for col in numeric_cols:
                try:
                    self._terminal_distribution_plot(
                        df[col].dropna(),
                        col_name=col,
                        width=terminal_width,
                        height=terminal_height,
                        bins=hist_bins
                    )
                except Exception as e:
                    self._print(f"[red]Failed to plot {col}: {str(e)}[/]")
            return None

    def _terminal_distribution_plot(
        self, 
        data: pd.Series, 
        col_name: str,
        width: int = 80,
        height: int = 20,
        bins: int = 20
    ) -> None:
        """Generate terminal-optimized distribution visualization."""
        try:
            import plotext as plt
        except ImportError:
            return self._text_histogram_fallback(data, col_name, width, bins)

        plt.clf()
        plt.subplots(1, 2)
        
        # Histogram
        plt.subplot(1, 1)
        plt.hist(data, bins=bins)
        plt.title(f"{col_name} Distribution")
        plt.plot_size(width//2, height)
        
        # Box plot
        plt.subplot(1, 2)
        plt.box_plot(data)
        plt.title(f"{col_name} Box Plot")
        plt.plot_size(width//2, height)
        
        plt.show()
        self._print("\n" + "-"*width + "\n")

    def _text_histogram_fallback(
        self, 
        data: pd.Series, 
        col_name: str, 
        width: int, 
        bins: int
    ) -> None:
        """ASCII histogram fallback when plotext is unavailable."""
        from numpy import histogram, linspace
        
        counts, edges = histogram(data, bins=bins)
        max_count = counts.max()
        
        self._print(f"\n[bold]{col_name} Distribution[/]")
        self._print(f"Records: {len(data):,} | Min: {data.min():.2f} | Max: {data.max():.2f}")
        
        for i in range(bins):
            bar_width = int((counts[i]/max_count) * (width-20)) if max_count > 0 else 0
            self._print(
                f"{edges[i]:>8.2f} - {edges[i+1]:<8.2f} | "
                f"[cyan]{'█'*bar_width}[/] {counts[i]:,}"
            )

    def _calculate_feature_stats(self, df: pd.DataFrame, col: str) -> Dict:
        """Calculate comprehensive statistics for a feature"""
        DECIMAL_PLACES = 4
        stats = {
            'Count': self._smart_round(df[col].count(), DECIMAL_PLACES),
            'Min': self._smart_round(df[col].min(), DECIMAL_PLACES),
            'Max': self._smart_round(df[col].max(), DECIMAL_PLACES),
            'Mean': self._smart_round(df[col].mean(), DECIMAL_PLACES),
            'Std Dev': self._smart_round(df[col].std(), DECIMAL_PLACES)
        }
        if col in self.param_limits:
            lsl, usl = self.param_limits[col]
            specs = {
                'LSL': lsl,
                'USL': usl,
                'Defects': ((df[col] < lsl) | (df[col] > usl)).sum(),
                '% Defects': round((((df[col] < lsl) | (df[col] > usl)).mean() * 100), 1)
            }
            
            # Process capability calculations
            sigma = df[col].std()
            if sigma > 0:
                pp = (usl - lsl) / (6 * sigma)
                ppu = (usl - df[col].mean()) / (3 * sigma)
                ppl = (df[col].mean() - lsl) / (3 * sigma)
                ppk = min(ppu, ppl)
                specs.update({'Pp': round(pp, 2), 'Ppk': round(ppk, 2)})
            
            stats.update(specs)

        return {k: self._smart_round(v, 4) for k, v in stats.items()}# if isinstance(v, float) else v for k, v in stats.items()}
    
    def analyze_correlations(self) -> Dict:
        """
        Calculate and visualize correlation matrices using multiple methods

        Returns:
            Dict: Contains:
                - pearson: Pearson correlation matrix (pd.DataFrame)
                - spearman: Spearman rank correlation matrix (pd.DataFrame)
                - kendall: Kendall's tau correlation matrix (pd.DataFrame)
                - plot: Interactive heatmap visualization (plotly Figure)

        Raises:
            ValueError: If dataset contains fewer than 2 numeric columns

        Example:
            >>> correlations = analyzer.analyze_correlations()
            >>> correlations['plot'].show()  # Display interactive heatmap
            >>> correlations['pearson']  # Access Pearson correlation matrix

        Note:
            Returns empty plot in case of visualization errors
            Preserves original correlation matrices even if plotting fails
        """
        numeric_df = self.production.select_dtypes(include=np.number)
        if len(numeric_df.columns) < 2:
            raise ValueError("Need at least 2 numeric columns for correlation analysis")

        corr_data = {
            'pearson': numeric_df.corr(),
            'spearman': numeric_df.corr(method='spearman'),
            'kendall': numeric_df.corr(method='kendall')
        }

        try:
            fig = px.imshow(corr_data['pearson'],
                        x=corr_data['pearson'].columns,
                        y=corr_data['pearson'].columns,
                        color_continuous_scale='RdBu_r',
                        zmin=-1,
                        zmax=1,
                        title="Pearson Correlation Matrix")
            fig.update_layout(
                width=1000, 
                height=800,
                margin=dict(l=100, r=100, t=50, b=100),
                xaxis=dict(tickfont=dict(size=12)),
                yaxis=dict(tickfont=dict(size=12))
            )
            corr_data['plot'] = fig
        except Exception as e:
            corr_data['plot'] = None
            print(f"Visualization error: {str(e)}")

        return corr_data
    
    def plot_interactive_timeline(self, parameters: List[str] = None) -> go.Figure:
        """
        Create interactive time series visualization with multiple parameters
        Parameters:
            parameters (List[str]): List of parameters to visualize. 
                                Default: all numeric parameters
        Returns:
            go.Figure: Interactive Plotly figure with:
                - Time series lines for selected parameters
                - Control limit annotations (if defined)
                - Range selector and hover tooltip
                - Multiple y-axis support
        Raises:
            ValueError: If date column not configured or invalid parameters specified
        """
        if not self.date_col:
            raise ValueError("Time series visualization requires date column configuration\nInitialize analyzer with date_col parameter")
        numeric_params = self.production.select_dtypes(include=np.number).columns.tolist()
        params = parameters or numeric_params
        if self.date_col in params:
            params.remove(self.date_col)
        invalid_params = set(params) - set(numeric_params)
        if invalid_params:
            raise ValueError(f"Non-numeric parameters cannot be plotted: {invalid_params}")
        df = self.production.set_index(self.date_col)
        fig = px.line(df, x=df.index, y=params, title="Production Parameters Timeline")
        
        # Added warning for missing control limits
        if not self.param_limits:
            self._print("[yellow]No control limits defined. Plotting raw data only.[/]")
        
        for param in params:
            if param in self.param_limits:
                lower, upper = self.param_limits[param]
                fig.add_hline(y=lower, line_dash="dot", line_color="red",
                            annotation_text=f"{param} LSL")
                fig.add_hline(y=upper, line_dash="dot", line_color="red",
                            annotation_text=f"{param} USL")
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Parameter Values",
            hovermode="x unified",
            legend_title="Parameters",
            xaxis_rangeslider_visible=True
        )
        return fig

    def create_interactive_dashboard(self) -> None:
        """
        Launch an interactive dashboard for data exploration

        Returns:
            pn.Column: Panel dashboard containing:
                - Parameter selection widget
                - Aggregation level control
                - Interactive timeline visualization
                - Correlation matrix selector

        Raises:
            RuntimeError: If not running in Jupyter environment

        Note:
            Requires Jupyter notebook/lab or Google Colab environment
            Call pn.extension() before using in notebook
        """
        try:
            import panel as pn
            from IPython.display import display, clear_output, HTML
            import nest_asyncio
            nest_asyncio.apply()
            import threading
            import time
        except ImportError as e:
            raise RuntimeError(f"Required packages missing: {str(e)}")

        # Singleton server instance check
        if hasattr(self, '_dashboard_server'):
            print(f"[!] Dashboard already running at: {self._dashboard_url}")
            return self._dashboard

        # Environment detection
        IN_COLAB = 'google.colab' in str(get_ipython())
        PORT = 43687

        try:
            # Colab-specific setup
            if IN_COLAB:
                # Install requirements once
                if not hasattr(self, '_colab_deps_installed'):
                    import subprocess
                    subprocess.run(["pip", "install", "-q", "jupyter_bokeh"], check=True)
                    self._colab_deps_installed = True
                    
                pn.extension(comms='colab', notifications=True)
            else:
                pn.extension()

            # Dashboard components (existing code)
            numeric_params = [
                col for col in self.production.select_dtypes(include=np.number).columns 
                if col != self.date_col
            ]

            # Widget creation
            param_selector = pn.widgets.MultiSelect(
                name="Parameters", options=numeric_params, size=8
            )
            aggregation_selector = pn.widgets.Select(
                name="Aggregation", options=['raw', 'hourly', 'daily', 'weekly'], width=200
            )
            corr_method_selector = pn.widgets.Select(
                name="Correlation Method", options=['pearson', 'spearman', 'kendall'], width=200
            )

            # Create reactive components
            @pn.depends(param_selector.param.value, aggregation_selector.param.value)
            def timeline_plot(params, agg):
                if agg != 'raw':
                    self.aggregate_data(period=agg)
                    df = self.daily_data
                else:
                    df = self.production
                return self.plot_interactive_timeline(parameters=params)

            @pn.depends(param_selector.param.value, corr_method_selector.param.value)
            def correlation_plot(params, method):
                if len(params) < 2:
                    return pn.pane.Markdown("Select at least 2 parameters for correlation")
                try:
                    corr_data = self.analyze_correlations()
                    return corr_data['plot']
                except ValueError as e:
                    return pn.pane.Alert(str(e), alert_type="warning")

            # Compose dashboard
            self._dashboard = pn.Column(
                pn.Row(
                    pn.Column(param_selector, aggregation_selector, corr_method_selector),
                    pn.Tabs(
                        ("Timeline", timeline_plot),
                        ("Correlations", correlation_plot)
                    )
                )
            )

            # Server launch
            if IN_COLAB:
                from panel.io.server import get_server
                from google.colab.output import eval_js
                
                # Create server with port configuration
                self._dashboard_server = get_server(
                    self._dashboard,
                    port=PORT,
                    allow_websocket_origin=['*'],
                    show=False
                )
                
                # Start server thread
                server_thread = threading.Thread(target=self._dashboard_server.start)
                server_thread.daemon = True
                server_thread.start()
                
                # Generate URL after brief delay
                time.sleep(1)
                self._dashboard_url = eval_js(f"google.colab.kernel.proxyPort({PORT})")
                display(HTML(f'<h3><a href="{self._dashboard_url}" target="_blank">Open Dashboard</a></h3>'))
                
            else:
                self._dashboard.show(port=PORT)
                
            return self._dashboard

        except Exception as e:
            clear_output()
            print(f"[!] Dashboard failed to initialize: {str(e)}")
            if IN_COLAB:
                print("Try: Runtime → Restart runtime → Run again")
            raise

    def visualize_eda_report(self, eda_report: Dict, output_format: str = 'html', output_path: str = 'eda_report.html') -> Optional[str]:
        """
        Transform EDA raw dawa into visual format

        Parameters:
            eda_report (Dict):      EDA data from generate_eda_report()
            output_format (str):    Output format: 'html' (default), 'console', 'notebook'
            output_path (str):      File path for saved report (html format only)

        Returns:
            str: HTML content if output_format='html' and output_path=None
            None: For other formats or when saving to file

        Raises:
            ValueError:     For invalid output formats
            RuntimeError:   If required visualization dependencies are missing

        Example:
            >>> report = analyzer.generate_eda_report()
            >>> analyzer.visualize_eda_report(report, output_format='notebook')
        """
        try:
            import matplotlib.pyplot as plt
            from jinja2 import Template
            import base64
        except ImportError as e:
            raise RuntimeError(f"Missing visualization dependencies: {str(e)}")

        report_data = {
            'summary': self._format_summary(eda_report['summary_stats']),
            'missing_data': self._format_missing_data(eda_report['missing_data']),
            'correlations': self._format_correlations(eda_report['correlations']),
            'plots': self._embed_visualizations(eda_report)
        }

        if output_format == 'html':
            return self._generate_html_report(report_data, output_path)
        elif output_format == 'console':
            self._print_console_report(report_data)
        elif output_format == 'notebook':
            self._display_notebook_report(report_data)
        else:
            raise ValueError(
                f"Invalid output format: {output_format}\n"
                "Choose from 'html', 'console', 'notebook'"
                )

    def _format_summary(self, summary_stats: pd.DataFrame) -> Dict:
        """Format numerical summary statistics for display"""
        return {
            'stats_table': summary_stats.style
                .format(lambda x: f"{self._smart_round(x, 4)}")
                .set_caption("Numerical summary statistics")
                .to_html(),
            'statistical_insights': self._generate_statistical_insights(summary_stats)
        }

    def _format_missing_data(self, missing_data: pd.DataFrame) -> Dict:
        """Format missing data analysis with visual indicators"""
        return {
            'missing_table': missing_data.style
                .bar(subset=['missing_pct'], color='#d65f5f')
                .format({'missing_pct': "{:.1%}"})
                .to_html(),
            'completeness_score': 1 - missing_data['missing_pct'].mean()
        }

    def _embed_visualizations(self, eda_report: Dict) -> Dict:
        """Convert plots to embeddable formats"""
        return {
            'distributions': self._plot_to_html(eda_report.get('distribution_plots')),
            'correlation_matrix': self._plot_to_html(
                eda_report.get('correlations', {}).get('plot')
            ),
            'temporal_trends': self._plot_to_html(eda_report.get('temporal_trends'))
        }

    def _plot_to_html(self, fig) -> str:
        """Convert matplotlib/plotly figure to HTML string"""
        from io import BytesIO
        if fig is None:
            return "<p>Visualization not available</p>"
        
        if 'plotly' in str(type(fig)):
            return fig.to_html(full_html=False)
        
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'
        
    def _generate_statistical_insights(self, summary_stats: pd.DataFrame) -> str:
        """
        Generate prioritized technical insights with problem indicators
        
        Parameters:
            summary_stats (pd.DataFrame): Statistical summary containing:
                - 'min', 'max': Absolute range values
                - '25%', '75%': Quartiles for IQR calculation
                - 'std', 'mean': Variability metrics
                - '50%': Median value
                - '1%', '99%': Extreme percentile values
                
        Returns:
            str: Formatted analysis containing:
                1. Critical issues requiring attention (outliers, high variability, skewness, control limit violations)
                2. Key metrics for each parameter (range, distribution shape, central tendency, process capability)
                
        Statistical Methods:
            - Outlier detection: Tukey's method (1.5×IQR rule)
            - Variability assessment: Coefficient of Variation (CV = σ/μ)
            - Skewness detection: 
                * Mean-Median comparison (>10% of mean)
                * Symmetry analysis (|Mean - Median| < 5% of mean)
            - Process capability analysis (Cp/Cpk) when control limits defined
                
        Example Output:
            [!] Critical Insights:
            • Temperature: 
                - Extreme upper values (max=98.3 exceeds 85.4 threshold)
                - High variability (CV=42%)
            • Pressure: 
                - USL violation (max=125.6 > 120.0)
            
            Key Metrics:
            • Vibration:
                - Range: 0.5-7.8 (Δ=7.3)
                - IQR: 1.2 | CV: 0.18
                - Central: Mean=4.2 (Median=4.1 ±1.5 σ)
                - Process capability: Cp=1.33, Cpk=1.25
            • RPM:
                - Constant values in 25-75% range
                - 25% values <850 (potential underperformance)
            
        Thresholds:
            - Outliers: Values beyond 1.5×IQR from quartiles
            - High variability: CV > 40%
            - Skewness: |Mean - Median| > 10% of mean
            - Process capability: Cp < 1.33 or Cpk < 1.0 indicates process issues
        """
        if summary_stats.empty:
            return "No numerical parameters available for statistical insights"
        
        critical = []
        metrics = []
        
        for param, stats in summary_stats.iterrows():
            # Derived metrics
            iqr = stats['75%'] - stats['25%']
            lower_threshold = stats['25%'] - 1.5 * iqr
            upper_threshold = stats['75%'] + 1.5 * iqr
            
            # Process capability calculations
            cp, cpk = None, None
            if param in self.param_limits:
                lsl, usl = self.param_limits[param]
                sigma = stats['std']
                if sigma > 0:
                    cp = (usl - lsl) / (6 * sigma)
                    cpk = min((usl - stats['mean']) / (3 * sigma),
                            (stats['mean'] - lsl) / (3 * sigma))
            
            # Critical conditions
            critical_notes = []
            if stats['max'] > upper_threshold:
                critical_notes.append(f"Extreme upper values (max={stats['max']:.1f} exceeds {upper_threshold:.1f} threshold)")
            if stats['min'] < lower_threshold:
                critical_notes.append(f"Extreme lower values (min={stats['min']:.1f} below {lower_threshold:.1f} threshold)")
            if stats['mean'] != 0 and (stats['std'] / stats['mean']) > 0.4:
                critical_notes.append(f"High variability (CV={stats['std']/stats['mean']:.0%})")
            if abs(stats['mean'] - stats['50%']) > 0.1 * stats['mean']:
                critical_notes.append(f"Skew (mean={stats['mean']:.1f} ≠ median={stats['50%']:.1f})")
            if param in self.param_limits:
                if stats['max'] > self.param_limits[param][1]:
                    critical_notes.append(f"USL violation (max={stats['max']:.1f} > {self.param_limits[param][1]:.1f})")
                if stats['min'] < self.param_limits[param][0]:
                    critical_notes.append(f"LSL violation (min={stats['min']:.1f} < {self.param_limits[param][0]:.1f})")
            
            # Metric details
            metric_notes = [
                f"Range: {stats['min']:.1f}-{stats['max']:.1f} (Δ={stats['max'] - stats['min']:.1f})",
                f"IQR: {iqr:.1f} | CV: {stats['std']/stats['mean']:.2f}" if stats['mean'] != 0 else "IQR: {iqr:.1f} | CV: N/A (zero mean)"
            ]
            
            # Central tendency with improved clarity
            central_tendency = (
                f"Central: Mean={stats['mean']:.1f} (Median={stats['50%']:.1f} ±{stats['std']:.1f} σ)"
            )
            metric_notes.append(central_tendency)
            
            # Process capability metrics
            if cp is not None:
                metric_notes.append(f"Capability: Cp={cp:.2f} | Cpk={cpk:.2f}")
            
            # Special cases
            if stats['25%'] == stats['75%']:
                metric_notes.append("Constant values in 25-75% range")
            if stats['mean'] != 0 and (stats['std'] / stats['mean']) < 0.1:
                metric_notes.append("Stable distribution (CV < 10%)")
            
            # Format outputs
            if critical_notes:
                critical.append(f"• {param}:\n  - " + "\n  - ".join(critical_notes))
            metrics.append(f"• {param}:\n  - " + "\n  - ".join(metric_notes))
        
        # Compose final output
        output = []
        if critical:
            output.append("[!] Critical Insights:")
            output.extend(critical)
            output.append("")
        
        output.append("Key Metrics:")
        output.extend(metrics)
        
        return '\n'.join(output)

    def _format_correlations(self, correlation_data: Dict) -> Dict:
        """
        Format correlation analysis results for visual reporting
        
        Parameters:
            correlation_data (Dict): Output from analyze_correlations()
            
        Returns:
            Dict: Formatted content with:
                - matrix_table: Styled correlation matrix HTML
                - top_positive: Strongest positive correlations
                - top_negative: Strongest negative correlations
                - heatmap_plot: Correlation heatmap visualization
        """
        if not correlation_data or 'matrix' not in correlation_data:
            return {
                'matrix_table': '<p>No correlation data available</p>',
                'insights': 'Insufficient numeric parameters for correlation analysis'
            }

        # Style correlation matrix table
        matrix = correlation_data['matrix']
        styled_matrix = matrix.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1)\
                                .format("{:.2f}")\
                                .set_caption("Correlation Matrix")
        
        # Generate statistical insights
        insights = self._generate_correlation_insights(matrix)
        
        return {
            'matrix_table': styled_matrix.to_html(),
            'insights': insights,
            'heatmap_plot': correlation_data.get('plot', None)
        }

    def _generate_correlation_insights(self, corr_matrix: pd.DataFrame) -> str:
        """
        Identify significant correlations from matrix
        
        Parameters:
            corr_matrix (pd.DataFrame): Square correlation matrix
            
        Returns:
            str: Formatted insights with top correlations
        
        Example:
            • Strong positive: Temp vs Pressure (r=0.89)
            • Strong negative: FlowRate vs Voltage (r=-0.78)
        """
        insights = []
        seen_pairs = set()
        
        # Flatten matrix and filter meaningful correlations
        pairs = corr_matrix.unstack().sort_values(ascending=False)
        for (param1, param2), value in pairs.items():
            if param1 == param2 or (param2, param1) in seen_pairs:
                continue
                
            if abs(value) > 0.7:  # Strong correlation threshold
                descriptor = "Strong positive" if value > 0 else "Strong negative"
                insights.append(f"• {descriptor}: {param1} vs {param2} (r={value:.2f})")
                seen_pairs.add((param1, param2))
                
            elif abs(value) > 0.5:  # Moderate correlation
                descriptor = "Moderate positive" if value > 0 else "Moderate negative"
                insights.append(f"• {descriptor}: {param1} vs {param2} (r={value:.2f})")
                seen_pairs.add((param1, param2))
                
        return '\n'.join(insights[:5])

    def _generate_html_report(self, report_data: dict, output_path: str) -> str:
        """
        Generate styled HTML report from EDA data
        
        Parameters:
            report_data (dict): Processed EDA components
            output_path (str): File path to save report
            
        Returns:
            str: HTML content if output_path=None
            
        Raises:
            RuntimeError: If template rendering fails
        """
        from jinja2 import Template
        import base64
        from datetime import datetime
        
        # HTML template with embedded styling
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Production Data Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .section { margin-bottom: 30px; border-bottom: 1px solid #eee; padding-bottom: 20px; }
                h2 { color: #2c3e50; }
                table { border-collapse: collapse; margin: 20px 0; }
                th, td { padding: 8px 12px; border: 1px solid #ddd; }
                th { background-color: #f8f9fa; }
                img { max-width: 100%; margin: 10px 0; }
                .alert { color: #856404; background-color: #fff3cd; padding: 10px; }
            </style>
        </head>
        <body>
            <h1>Production Data Analysis Report</h1>
            <p>Generated: {{ timestamp }}</p>
            
            <div class="section">
                <h2>Summary Statistics</h2>
                {{ summary.stats_table|safe }}
            </div>

            <div class="section">
                <h2>Statistical Insights</h2>
                <pre>{{ summary.statistical_insights }}</pre>
            </div>
            
            <div class="section">
                <h2>Missing Data Analysis</h2>
                {{ missing_data.missing_table|safe }}
            </div>
            
            <div class="section">
                <h2>Feature Distributions</h2>
                {{ plots.distributions|safe }}
            </div>
            
            <div class="section">
                <h2>Correlation Analysis</h2>
                {{ plots.correlation_matrix|safe }}
            </div>

            <div class="section">
                <h2>Correlation Insights</h2>
                <pre>{{ correlations.insights }}</pre>
            </div>
            
            <div class="section">
                <h2>Temporal Trends</h2>
                {{ plots.temporal_trends|safe }}
            </div>
        </body>
        </html>
        """
        
        try:
            # Render template with data
            template = Template(html_template)
            html_content = template.render(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                **report_data
            )
            
            # Save or return HTML
            if output_path:
                with open(output_path, 'w') as f:
                    f.write(html_content)
                return f"Report saved to {output_path}"
            return html_content
            
        except Exception as e:
            raise RuntimeError(f"HTML report generation failed: {str(e)}")

    def _display_notebook_report(self, report_data: dict) -> None:
        """
        Display interactive EDA report in Jupyter notebooks
        
        Features:
        - Integrated HTML display
        - Interactive plot preservation
        - Responsive layout
        """
        from IPython.display import display, HTML
        
        # Create report sections
        sections = [
            ("Summary Statistics", report_data['summary']['stats_table']),
            ("Statistical Insights", f"<pre>{report_data['summary']['statistical_insights']}</pre>"),
            ("Missing Data Analysis", report_data['missing_data']['missing_table']),
            ("Feature Distributions", report_data['plots']['distributions']),
            ("Correlation Matrix", report_data['plots']['correlation_matrix']),
            ("Correlation Insights", f"<pre>{report_data['correlations']['insights']}</pre>"),
            ("Temporal Trends", report_data['plots']['temporal_trends'])
        ]
        
        # Display each section with styling
        display(HTML("<h1 style='color: #2c3e50'>Production Data Analysis Report</h1>"))
        
        for title, content in sections:
            display(HTML(
                f"<div style='margin: 20px 0; border-bottom: 2px solid #eee; padding-bottom: 20px'>"
                f"<h2 style='color: #34495e'>{title}</h2>"
                f"{content}"
                f"</div>"
            ))
