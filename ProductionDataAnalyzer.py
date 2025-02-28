import os
import io
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
from google.colab import files
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

    3. **Machine Learning & Advanced Analytics**
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
    def _optimize_dtypes(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage with safe numeric conversion

        Parameters:
            df (pd.DataFrame): DataFrame to optimize
            date_col (str):    Name of the date column

        Returns:
            pd.DataFrame: Optimized DataFrame
        """
        df = df.copy()

        for col in df.columns:
            # Convert the date column to datetime using the specific format
            if col == date_col:
                df[date_col] = pd.to_datetime(
                    df[date_col],
                    format='%d.%m.%Y %H:%M',
                    errors='coerce'
                )
                continue

            # Phase 1: numeric conversion
            if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                # Numeric conversion preserving text
                numeric_vals = pd.to_numeric(df[col], errors='coerce')
                if numeric_vals.notnull().mean() > 0.95:
                    df[col] = numeric_vals.astype(pd.to_numeric(df[col], downcast='float').dtype)
                else:
                    # Only convert to category if not convertible
                    unique_ratio = df[col].nunique() / len(df)
                    if unique_ratio < 0.1:
                        df[col] = df[col].astype('category')

            # Phase 2: downcast numerics
            elif pd.api.types.is_numeric_dtype(df[col]):
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

    def save_aggregated_data(self, filename: str = 'aggregated_data.csv') -> None
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
                "Check directory permissions or try different location
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
