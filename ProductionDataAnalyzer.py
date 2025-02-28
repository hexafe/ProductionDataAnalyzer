import os
import io
import tempfile
from pathlib import Path
import pandas as pd
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
            'dtype': 'object',
            'parse_dates': bool(date_col),
            'dayfirst': False,
            'na_values': ['\\N', '']
        }
        if date_col:
            default_csv_kwargs['parse_dates'] = [date_col]
            default_csv_kwargs['date_parser'] = lambda x: pd.to_datetime(
                x, format='ISO8601', errors='coerce'
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
                format='ISO8601',
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
        for col in df.columns:
            # Convert the date column to datetime using the specific format
            if col == date_col:
                df[date_col] = pd.to_datetime(
                    df[date_col],
                    format='ISO8601',
                    errors='coerce'
                )
                continue

            # Attempt conversion to numeric
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Only update if conversion yields valid numbers
                if df[col].notna().mean() > 0.8:
                    df[col] = pd.to_numeric(df[col], downcast='float')
            except:
                pass

            # Convert object columns with low cardinality to categorical
            if df[col].dtype == 'object':
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.1:
                    df[col] = df[col].astype('category')

        return df

    @staticmethod
    def filter_by_id(production_data_df: pd.DataFrame, id_data_df: pd.DataFrame, id_col: str) -> pd.DataFrame:
        """
        Filter DataFrame by matching IDs from another DataFrame

        Parameters:
            df (pd.DataFrame):    DataFrame to filter
            df_id (pd.DataFrame): DataFrame containing IDs to match
            id_col (str):         Name of the ID column in both DataFrames

        Returns:
            pd.DataFrame: Filtered DataFrame
        """
        # Get the IDs
        id_list = id_data_df[id_col]

        # Create a filter based on IDs
        filter = production_data_df[id_col].isin(id_list)

        return production_data_df[filter]

    @staticmethod
    def save_to_csv(df: pd.DataFrame,
                    filename: str = 'output.csv',
                    sep: str = ';',
                    decimal: str = ','
    ) -> None:
        """
        Save DataFrame to CSV file

        Parameters:
            df (pd.DataFrame): DataFrame to save
            filename (str):    Name of the CSV file to save to
            sep (str):         Separator used in the CSV file
            decimal (str):     Decimal separator used in the CSV file

        Returns:
            None
        """
        df.to_csv(filename, sep=sep, decimal=decimal, index=False)
