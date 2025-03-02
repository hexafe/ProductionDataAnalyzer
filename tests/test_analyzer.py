# tests/test_analyzer.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock, call
from pathlib import Path
import tempfile
import shutil
from ProductionDataAnalyzer.analyzer import ProductionDataAnalyzer

# Fixtures ----------------------------------------------------------------

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=100, freq='H'),
        'temperature': np.random.normal(50, 5, 100),
        'pressure': np.random.uniform(80, 120, 100),
        'part_id': [f"ID_{i:03d}" for i in range(100)],
        'status': np.random.choice(['OK', 'FAIL'], 100, p=[0.95, 0.05])
    })

@pytest.fixture
def analyzer(sample_data):
    return ProductionDataAnalyzer(sample_data, date_col='timestamp')

@pytest.fixture
def limits_dict():
    return {'temperature': (40, 60), 'pressure': (90, 110)}

# Core Functionality Tests -------------------------------------------------

class TestInitialization:
    def test_valid_initialization(self, sample_data):
        analyzer = ProductionDataAnalyzer(sample_data, 'timestamp')
        assert analyzer.date_col == 'timestamp'
        assert len(analyzer.production) == 100
        assert 'temperature' in analyzer.selected_params

    def test_invalid_data_type(self):
        with pytest.raises(TypeError):
            ProductionDataAnalyzer([1, 2, 3], 'timestamp')

    def test_missing_date_column(self, sample_data):
        with pytest.raises(ValueError):
            ProductionDataAnalyzer(sample_data, 'invalid_column')

    def test_non_datetime_column(self):
        df = pd.DataFrame({'invalid_date': ['2023-01-01', '2023-01-02']})
        with pytest.raises(ValueError):
            ProductionDataAnalyzer(df, 'invalid_date')

# Data Ingestion Tests -----------------------------------------------------

class TestFileUpload:
    @patch('google.colab.files.upload')
    @patch('pyunpack.Archive')
    def test_upload_process(self, mock_archive, mock_upload, tmp_path):
        # Setup mock file upload
        test_files = {
            'data.zip': b'zip_content',
            'data.csv': b'timestamp;value\n2023-01-01 12:00;25',
            'data.xlsx': b'excel_content'
        }
        mock_upload.return_value = test_files
        
        # Mock archive extraction
        mock_archive.return_value.extractall.side_effect = lambda x: (
            (tmp_path / 'extracted.csv').touch()
        )

        df = ProductionDataAnalyzer.upload_files(
            date_col='timestamp',
            tmp_dir=str(tmp_path),
            remove_tmp_dir=False
        )

        assert not df.empty
        assert 'timestamp' in df.columns
        assert mock_archive.called

    def test_post_merge_cleanup(self, sample_data):
        duplicated = pd.concat([sample_data, sample_data])
        cleaned = ProductionDataAnalyzer._post_merge_cleanup(
            duplicated, 'timestamp'
        )
        assert len(cleaned) == len(sample_data)
        assert pd.api.types.is_datetime64_any_dtype(cleaned['timestamp'])

    def test_dtype_optimization(self):
        test_df = pd.DataFrame({
            'str_num': ['1', '2', '3'],
            'category': ['A', 'A', 'B'],
            'float': [1.1, 2.2, 3.3]
        })
        optimized = ProductionDataAnalyzer._optimize_dtypes(test_df, None)
        assert pd.api.types.is_integer_dtype(optimized['str_num'])
        assert pd.api.types.is_categorical_dtype(optimized['category'])

# Data Processing Tests ----------------------------------------------------

class TestDataFiltering:
    def test_valid_filtering(self, sample_data):
        reference_ids = pd.DataFrame({'part_id': ['ID_001', 'ID_002']})
        filtered = ProductionDataAnalyzer.filter_by_id(
            sample_data, reference_ids, 'part_id'
        )
        assert len(filtered) == 2
        assert filtered['part_id'].isin(['ID_001', 'ID_002']).all()

    def test_no_matches(self, sample_data):
        reference_ids = pd.DataFrame({'part_id': ['INVALID_ID']})
        with pytest.raises(ValueError):
            ProductionDataAnalyzer.filter_by_id(
                sample_data, reference_ids, 'part_id'
            )

    def test_missing_id_column(self, sample_data):
        with pytest.raises(KeyError):
            ProductionDataAnalyzer.filter_by_id(
                sample_data, sample_data, 'missing_col'
            )

# Temporal Analysis Tests --------------------------------------------------

class TestAggregation:
    def test_daily_aggregation(self, analyzer):
        analyzer.aggregate_data('day')
        assert not analyzer.daily_data.empty
        assert 'temperature' in analyzer.selected_params
        assert len(analyzer.daily_data) <= 5  # 100 hours = ~4.16 days

    def test_invalid_period(self, analyzer):
        with pytest.raises(ValueError):
            analyzer.aggregate_data('invalid_period')

    def test_save_aggregated_data(self, analyzer, tmp_path):
        analyzer.aggregate_data('day')
        test_file = tmp_path / 'test.csv'
        analyzer.save_aggregated_data(str(test_file))
        assert test_file.exists()
        assert test_file.stat().st_size > 0

# Parameter Limits Tests ---------------------------------------------------

class TestParameterLimits:
    def test_dict_limits(self, analyzer, limits_dict):
        analyzer.set_parameter_limits(limits_dict)
        assert analyzer.param_limits == limits_dict

    @patch('gspread.authorize')
    @patch('google.auth.default')
    def test_gsheet_limits(self, mock_auth, mock_gsheet, analyzer):
        # Mock Google Sheets response
        mock_sheet = MagicMock()
        mock_sheet.get_all_records.return_value = [
            {'parameter': 'temperature', 'LSL': 40, 'USL': 60},
            {'parameter': 'pressure', 'LSL': 90, 'USL': 110}
        ]
        mock_gsheet.return_value.open_by_url.return_value.get_worksheet.return_value = mock_sheet
        
        analyzer.set_parameter_limits(
            'https://docs.google.com/spreadsheets/d/test'
        )
        assert 'temperature' in analyzer.param_limits

    def test_dataframe_limits(self, analyzer, sample_data):
        limits_df = pd.DataFrame({
            'parameter': ['temperature', 'pressure'],
            'LSL': [40, 90],
            'USL': [60, 110]
        })
        analyzer.set_parameter_limits(limits_df)
        assert len(analyzer.param_limits) == 2

    def test_invalid_limit_source(self, analyzer):
        with pytest.raises(ValueError):
            analyzer.set_parameter_limits(12345)

# Edge Case Tests ----------------------------------------------------------

class TestEdgeCases:
    def test_empty_dataframe(self):
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError):
            ProductionDataAnalyzer.filter_by_id(
                empty_df, pd.DataFrame({'id': [1]}), 'id'
            )

    def test_all_null_dates(self):
        df = pd.DataFrame({
            'timestamp': [np.nan]*10,
            'value': range(10)
        })
        with pytest.raises(ValueError):
            ProductionDataAnalyzer(df, 'timestamp')

    def test_corrupted_archive(self, tmp_path):
        test_file = tmp_path / 'corrupted.zip'
        test_file.write_bytes(b'invalid_content')
        
        with pytest.raises(Exception):
            ProductionDataAnalyzer.upload_files(
                tmp_dir=str(tmp_path),
                archive_ext=('.zip',)
            )

# Utility Tests ------------------------------------------------------------

class TestUtilities:
    def test_csv_save(self, sample_data, tmp_path):
        test_file = tmp_path / 'test.csv'
        ProductionDataAnalyzer.save_to_csv(
            sample_data, str(test_file))
        assert test_file.exists()
        loaded = pd.read_csv(test_file, sep=';')
        assert not loaded.empty

    def test_invalid_csv_params(self):
        with pytest.raises(ValueError):
            ProductionDataAnalyzer.save_to_csv(
                pd.DataFrame(), 'invalid.txt'
            )

# Concurrency Tests --------------------------------------------------------

class TestConcurrency:
    @pytest.mark.parametrize("workers", [2, 4])
    def test_parallel_processing(self, workers):
        # Requires implementation of parallel processing in the class
        pass  # Placeholder for actual parallel processing tests

# Main ---------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main(["-v", "--cov=ProductionDataAnalyzer", "--cov-report=html"])