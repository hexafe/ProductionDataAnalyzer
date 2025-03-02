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
    # Update the test_upload_process method
    @patch('google.colab.files.upload', return_value={'test.zip': b'content'})
    @patch('pyunpack.Archive')
    @patch('google.colab.files.download')
    def test_upload_process(self, mock_download, mock_archive, mock_upload, tmp_path):
        # Mock extraction and create dummy extracted files
        mock_archive.return_value.extractall.side_effect = lambda x: (
            (tmp_path / 'extracted.csv').write_text('timestamp,temperature\n2023-01-01,50')
        )
        
        result = ProductionDataAnalyzer.upload_files(tmp_dir=str(tmp_path))
        
        mock_archive.assert_called_once_with(str(tmp_path / 'test.zip'))
        assert len(result) == 1
        assert 'extracted.csv' in str(result[0])

    def test_post_merge_cleanup(self, sample_data):
        duplicated = pd.concat([sample_data, sample_data])
        cleaned = ProductionDataAnalyzer._post_merge_cleanup(
            duplicated, 'timestamp'
        )
        assert len(cleaned) == len(sample_data)
        assert pd.api.types.is_datetime64_any_dtype(cleaned['timestamp'])

    def test_dtype_optimization(self):
        test_df = pd.DataFrame({
            'str_num': ['1', '1', '1'],
            'category': ['A', 'A', 'A'],
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

    @patch('pandas.DataFrame.to_csv')
    @patch('os.path.getsize', return_value=1024)
    def test_save_aggregated_data(self, mock_to_csv, analyzer):
        analyzer.aggregate_data('day')
        test_path = 'test.csv'
        analyzer.save_aggregated_data(test_path)
        mock_to_csv.assert_called_once_with(test_path, sep=';', index=False)

# Parameter Limits Tests ---------------------------------------------------

class TestParameterLimits:
    def test_dict_limits(self, analyzer, limits_dict):
        analyzer.set_parameter_limits(limits_dict)
        assert analyzer.param_limits == limits_dict

    @patch('gspread.authorize')
    @patch('google.auth.default')
    @patch('google.colab.files.download')
    def test_gsheet_limits(self, mock_download, mock_auth, mock_gsheet, analyzer):
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
        assert analyzer.param_limits == {
            'temperature': (40, 60),
            'pressure': (90, 110)
        }

    def test_dataframe_limits(self, analyzer, sample_data):
        limits_df = pd.DataFrame({
            'parameter': ['temperature', 'pressure'],
            'LSL': [40, 90],
            'USL': [60, 110]
        })
        analyzer.set_parameter_limits(limits_df)
        assert len(analyzer.param_limits) == 2

    def test_invalid_limit_source(self, analyzer):
        with pytest.raises(RuntimeError):
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
            'timestamp': [pd.NaT]*10,
            'value': range(10)
        })
        with pytest.raises(ValueError):
            ProductionDataAnalyzer(df, 'timestamp')

    @patch('google.colab.files.upload', return_value={'corrupted.zip': b'invalid'})
    def test_corrupted_archive(self, mock_upload, tmp_path):
        with patch('pyunpack.Archive') as mock_archive:
            mock_archive.return_value.extractall.side_effect = Exception("Corrupted archive")
            
            with pytest.raises(Exception) as exc_info:
                ProductionDataAnalyzer.upload_files(
                    tmp_dir=str(tmp_path),
                    archive_ext=('.zip',)
                )
            # Check for the original error message
            assert "No valid data files processed" in str(exc_info.value)
            assert "Corrupted archive" in str(exc_info.value.__cause__)

# Utility Tests ------------------------------------------------------------

class TestUtilities:
    @patch('pandas.DataFrame.to_csv')
    @patch('os.path.getsize', return_value=1024)
    def test_csv_save(self, mock_to_csv, sample_data):
        test_path = 'test.csv'
        ProductionDataAnalyzer.save_to_csv(sample_data, test_path)
        mock_to_csv.assert_called_once_with(test_path, sep=';', index=False)

    def test_invalid_csv_params(self):
        with pytest.raises(ValueError):
            ProductionDataAnalyzer.save_to_csv(
                pd.DataFrame(), 'invalid.txt'
            )

# Main ---------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main(["-v", "--cov=ProductionDataAnalyzer", "--cov-report=html"])
