from typing import Union, Dict
import os
import io
import tempfile
from pathlib import Path
import pandas as pd
import matplotlib.dates as mdates
import gspread
from google.colab import files, auth
from google.auth import default
from pyunpack import Archive
import shutil

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
    def __init__(self, production_data: pd.DataFrame, date_col: str = None):
        """
        Initialize a production data analyzer with raw data and configuration

        Parameters:
            production_data (pd.DataFrame): Input data containing production records
                                            Must contain a datetime column and various parameters to analyze
            date_col (str):                 Name of the datetime column used for temporal analysis

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
            if not pd.api.types.is_datetime64_any_dtype(production_data[date_col]):
                raise ValueError(f"Column '{date_col}' must be datetime type")

        # Core attributes
        self.date_col = date_col
        self.production = production_data
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

        # Initialization feedback
        print(f"Analyzer initialized with {len(self.production):,} records ({len(self.selected_params)} parameters)")

    @staticmethod
    def upload_files(
        date_col: str = None,
        csv_kwargs: dict = None,
        excel_kwargs: dict = None,
        archive_ext: tuple = ('.zip', '.7z', '.rar', '.tar', '.gz'),
        csv_ext: tuple = ('.csv',),
        excel_ext: tuple = ('.xls', '.xlsx'),
        tmp_dir: str = '/content/tmp_upload',
        remove_tmp_dir: bool = True,
        chunksize: int = 100000,
    ) -> pd.DataFrame:
        """
        Upload files (CSV, Excel, and various archives) from Google Colab and return a combined DataFrame

        Custom keyword arguments for reading CSV and Excel files can be provided via
        `csv_kwargs` and `excel_kwargs`, respectively. Defaults are used if not specified

        Parameters:
            date_col (str):         Name of the column to parse as datetime
            csv_kwargs (dict):      Optional parameters for pd.read_csv
                                    Default: {'sep': ';', 'decimal': ',', 'parse_dates': [date_col],
                                              'dayfirst': True, 'na_values': ['\\N', '']}
            excel_kwargs (dict):    Optional parameters for pd.read_excel
                                    Default: {'engine': 'openpyxl'}
            archive_ext (tuple):    File extensions recognized as archives
                                    Default: ('.zip', '.7z', '.rar', '.tar', '.gz')
            csv_ext (tuple):        File extensions recognized as CSV files
                                    Default: ('.csv',)
            excel_ext (tuple):      File extensions recognized as Excel files
                                    Default: ('.xls', '.xlsx')
            tmp_dir (str):          Directory to store temporary files
                                    Default: '/content/tmp_upload'
            remove_tmp_dir (bool):  Whether to remove the temporary directory after processing
                                    Default: True
            chunksize (int):        Number of rows per chunk for processing large CSV files
                                    If None, CSV files are read in one go
                                    Default: 100000

        Returns:
            pd.DataFrame: Combined DataFrame after processing all uploaded files

        Raises:
            ValueError: If no valid data files are processed
        """
        # Merge user-provided CSV options with defaults (user values override defaults)
        default_csv_kwargs = {
            'sep': ';',
            'decimal': ',',
            'parse_dates': bool(date_col),
            'dayfirst': False,
            'na_values': ['\\N', '']
        }
        if date_col:
            default_csv_kwargs['parse_dates'] = [date_col]
            default_csv_kwargs['date_parser'] = lambda x: pd.to_datetime(
                x, format='%d.%m.%Y %H:%M', errors='coerce'
            )

        csv_kwargs = {**default_csv_kwargs, **(csv_kwargs or {})}

        # Merge user-provided Excel options with defaults
        default_excel_kwargs = {'engine': 'openpyxl'}
        excel_kwargs = {**default_excel_kwargs, **(excel_kwargs or {})}

        uploaded = files.upload()
        dfs = []
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_dir_path = Path(tmp_dir)

        try:
            # Process each uploaded file
            for fn, content in uploaded.items():
                try:
                    file_path = tmp_dir_path / fn
                    with open(file_path, 'wb') as f:
                        f.write(content)
                    lower_fn = fn.lower()
                    # Process archive files by extracting them
                    if any(lower_fn.endswith(ext) for ext in archive_ext):
                        try:
                            Archive(str(file_path)).extractall(str(tmp_dir_path))
                            print(f"Successfully extracted: {fn}")
                            # Remove the archive file after extraction
                            file_path.unlink()
                        except Exception as e:
                            print(f"Failed to extract {fn}: {str(e)}")
                            continue
                    # Process CSV files
                    elif any(lower_fn.endswith(ext) for ext in csv_ext):
                        if chunksize is None:
                            df = pd.read_csv(io.BytesIO(content), **csv_kwargs)
                        else:
                            chunks = pd.read_csv(io.BytesIO(content), chunksize=chunksize, **csv_kwargs)
                            df = pd.concat(chunks, ignore_index=True)
                        dfs.append(df)
                        print(f"Processed CSV directly: {fn}")
                    # Process Excel files
                    elif any(lower_fn.endswith(ext) for ext in excel_ext):
                        df = pd.read_excel(io.BytesIO(content), **excel_kwargs)
                        dfs.append(df)
                        print(f"Processed Excel directly: {fn}")
                except Exception as e:
                    print(f"Error processing {fn}: {str(e)}")
                    continue

            # Process files extracted from archives
            for extracted_file in tmp_dir_path.rglob('*'):
                if extracted_file.is_file():
                    try:
                        lower_suffix = extracted_file.suffix.lower()
                        if lower_suffix in csv_ext:
                            if chunksize is None:
                                df = pd.read_csv(extracted_file, **csv_kwargs)
                            else:
                                chunks = pd.read_csv(extracted_file, chunksize=chunksize, **csv_kwargs)
                                df = pd.concat(chunks, ignore_index=True)
                            dfs.append(df)
                            print(f"Processed extracted CSV: {extracted_file.name}")
                        elif lower_suffix in excel_ext:
                            df = pd.read_excel(extracted_file, **excel_kwargs)
                            dfs.append(df)
                            print(f"Processed extracted Excel: {extracted_file.name}")
                    except Exception as e:
                        print(f"Error processing extracted file {extracted_file.name}: {str(e)}")
                        continue
        finally:
            # Clean up the temporary directory if requested
            if remove_tmp_dir:
                shutil.rmtree(tmp_dir_path, ignore_errors=True)

        if not dfs:
            raise ValueError(
                "No valid data files processed. Please check:\n"
                "1. File extensions (.csv, .xls, .xlsx)\n"
                "2. Archive integrity\n"
                "3. Supported formats: CSV, Excel, ZIP, 7z, RAR, TAR, GZ"
            )

        combined = pd.concat(dfs, ignore_index=True)
        return ProductionDataAnalyzer._post_merge_cleanup(combined, date_col)

    @staticmethod
    def _post_merge_cleanup(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """
        Perform cleanup on the combined DataFrame:
          - Remove duplicate rows
          - Sort by the date column
          - Reset index
          - Optimize data types to reduce memory usage

        Parameters:
            df (pd.DataFrame): Combined DataFrame
            date_col (str):    Name of the date column

        Returns:
            pd.DataFrame: Cleaned and optimized DataFrame
        """
        # Remove duplicate rows, keeping the first occurence
        df = df.drop_duplicates(keep='first')
        if date_col in df.columns:
            # Ensure the date column is parsed as datetime
            df[date_col] = pd.to_datetime(
                df[date_col],
                format='%d.%m.%Y %H:%M',
                errors='coerce'
            )
            # Sort by the date column and reset index
            df = df.sort_values(date_col).reset_index(drop=True)
        # Optimize data types for memory efficiency
        return ProductionDataAnalyzer._optimize_dtypes(df, date_col)

    @staticmethod
    def _optimize_dtypes(
        df: pd.DataFrame,
        date_col: str,
        numeric_threshold: float = 0.95,
        categorical_threshold: float = 0.1
    ) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage with safe numeric conversion

        Parameters:
            df (pd.DataFrame):              DataFrame to optimize
            date_col (str):                 Name of the date column
            numeric_threshold (float):      Proportion of unique values to consider numeric
            categorical_threshold (float):  Proportion of unique values to consider categorical

        Returns:
            pd.DataFrame: Optimized DataFrame
        """
        # Work on a copy to avoid modifying the original DataFrame
        df = df.copy()

        for col in df.columns:
            # Process the date column
            if col == date_col:
                df[col] = pd.to_datetime(
                    df[col],
                    format='%d.%m.%Y %H:%M',
                    errors='coerce'
                )
                continue

            # Process object/string columns
            if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                numeric_vals = pd.to_numeric(df[col], errors='coerce')
                valid_ratio = numeric_vals.notnull().mean()
                if valid_ratio >= numeric_threshold:
                    if numeric_vals.dropna().apply(float.is_integer).all():
                        df[col] = pd.to_numeric(numeric_vals, downcast='integer')
                    else:
                        df[col] = pd.to_numeric(numeric_vals, downcast='float')
                else:
                    unique_ratio = df[col].nunique() / len(df)
                    if unique_ratio <= categorical_threshold:
                        df[col] = df[col].astype('category')
            elif pd.api.types.is_numeric_dtype(df[col]):
                if pd.api.types.is_integer_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], downcast='integer')
                else:
                    df[col] = pd.to_numeric(df[col], downcast='float')

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

    @staticmethod
    def save_to_csv(
        df: pd.DataFrame,
        filename: str = 'output.csv',
        sep: str = ';',
        decimal: str = ','
    ) -> None:
        """
        Save a DataFrame to CSV file with configurable formatting and automatic download in Colab

        Parameters:
            df (pd.DataFrame):  DataFrame to save. Must contain valid column names and data
            filename (str):     Output file name/path. Must have .csv extension
                                Default: 'output.csv'
            sep (str):          Column separator. Recommended: ';' for European format, ',' for US.
                                Default: ';'
            decimal (str):      Decimal separator. Recommended: ',' for European, '.' for US.
                                Default: ','

        Returns:
            None: Outputs file to disk and triggers browser download in Colab environments

        Raises:
            ValueError:       If input validation fails for parameters or DataFrame
            PermissionError:  If file write permissions are insufficient
            RuntimeError:     For Colab-specific download failures

        Example:
            >>> df = pd.DataFrame({'temp': [20.5, 21.3], 'pressure': [101.3, 102.4]})
            >>> ProductionDataAnalyzer.save_to_csv(df, 'sensor_data.csv')
            Successfully saved 2 rows with 2 columns to sensor_data.csv (0.5KB)
            Downloading sensor_data.csv...
        """
        # Input validation
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Input must be a non-empty pandas DataFrame")
        
        if not isinstance(filename, str) or not filename.endswith('.csv'):
            raise ValueError("Filename must be string ending with .csv")

        if len(sep) != 1 or len(decimal) != 1:
            raise ValueError("Separators must be single-character strings")

        try:
            # Save with European-style formatting by default
            df.to_csv(
                filename,
                sep=sep,
                decimal=decimal,
                index=False,
                encoding='utf-8',
                date_format='%Y-%m-%d %H:%M:%S'
            )
        except PermissionError as pe:
            raise PermissionError(
                f"Write permission denied for {filename}. "
                "Check directory permissions or try different location."
            ) from pe
        except Exception as e:
            raise RuntimeError(f"CSV save failed: {str(e)}") from e

        # Generate output summary
        row_count = len(df)
        col_count = len(df.columns)
        file_size_kb = os.path.getsize(filename) / 1024
        
        print(
            f"Successfully saved {row_count} rows with {col_count} columns "
            f"to {filename} ({file_size_kb:.1f}KB)"
        )

        # Handle Colab-specific download
        try:
            from google.colab import files
            files.download(filename)
            print(f"Download initiated for {filename}")
        except ImportError:
            print(f"File saved locally at {os.path.abspath(filename)}")
        except Exception as e:
            raise RuntimeError(
                f"Download failed: {str(e)}. "
                f"File remains available at {os.path.abspath(filename)}"
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
        if period not in period_map:
            valid_options = list(period_map.keys())
            raise ValueError(f"Invalid period. Valid options: {valid_options}")

        # Check datetime column configuration
        if not self.date_col:
            raise ValueError("Temporal aggregation required date_col initialization")

        # Verify datetime column integrity
        date_series = self.production[self.date_col]
        if not pd.api.types.is_datetime64_any_dtype(date_series):
            raise ValueError(f"Column '{self.date_col}' must be datetime type")
        if date_series.is_monotonic_increasing is False:
            raise ValueError(f"Column '{self.date_col}' must be sorted in ascending order")

        # Identify numeric parameters for aggregation
        numeric_cols = self.production.select_dtypes(include=np.number).columns
        temporal_features = ['year', 'month', 'week', 'day']
        numeric_params = [col for col in numeric_cols if col not in temporal_features]

        if not numeric_params:
            raise ValueError(f"No aggregatalbe numeric parameters found\nExcluded temporal features: {temporal_features}")

        try:
            self.daily_data = (
                self.production
                .set_index(self.date_col)
                [numeric_params]
                .resample(period_map[period])
                .mean()
                .reset_index()
            )
        except pd.error.OutOfBoundsDatetime:
            raise ValueError("Datetime values outside pandas' supported range (1677-2262)")

        self.selected_params = [
            col for col in self.daily_data.columns
            if col != self.date_col and pd.api.types.is_numeric_dtype(self.daily_data[col])
        ]

        # Store frequency configuration
        self.agg_freq = period_map[period]
        self.period = period

        # Output status summary
        row_count = len(self.daily_data)
        param_count = len(self.selected_params)
        print(f"Aggregated to {row_count} {period} intervals with {param_count} parameters")

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
            gc = gspread.authorize(cred)

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
        
        except gspread.AuthenticationError as e:
            raise RuntimeError(
                "Google Sheets authentication blocked. Possible reasons:\n"
                "- Corporate firewall/settings restrictions\n"
                "- Missing required permissions\n"
                "- Invalid credentials\n"
                "Try CSV fallback method instead"
            ) from e

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
            return {
                df.set_index('parameter')
                [['LSL', 'USL']]
                .apply(tuple, axis=1)
                .to_dict()
            }

        except KeyError as ke:
            raise RecursionError(f"Column access error: {str(ke)}") from ke
        except pd.errors.ParserError as pe:
            raise ValueError(f"Data parsing failed: {str(pe)}") from pe
        except Exception as e:
            raise RuntimeError(f"Data processing failed: {str(e)}") from e

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
