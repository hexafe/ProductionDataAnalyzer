import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
from ProductionDataAnalyzer import ProductionDataAnalyzer
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

# Fixtures with improved data isolation
@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='D'),
        'temperature': np.random.normal(70, 5, 100),
        'pressure': np.random.uniform(80, 120, 100),
        'serial_number': [f'SN{i:04d}' for i in range(100)]
    }).astype({'temperature': 'float32', 'pressure': 'float32'})

@pytest.fixture
def invalid_data():
    return pd.DataFrame({
        'timestamp': ['invalid'] * 5,
        'value': [1, 2, 3, 4, 5]
    })

# Mock environment detection
@pytest.fixture(autouse=True)
def mock_environment(monkeypatch):
    monkeypatch.setattr('ProductionDataAnalyzer.IN_COLAB', False)
    monkeypatch.setattr('ProductionDataAnalyzer.gspread', MagicMock())

# Test Initialization
class TestInitialization:
    def test_valid_initialization(self, sample_data):
        analyzer = ProductionDataAnalyzer(sample_data, date_col='timestamp')
        assert analyzer.date_col == 'timestamp'
        assert len(analyzer.production) == 100

    def test_invalid_dataframe(self):
        with pytest.raises(TypeError):
            ProductionDataAnalyzer("not a dataframe")

    def test_missing_date_column(self, sample_data):
        with pytest.raises(ValueError):
            ProductionDataAnalyzer(sample_data, date_col='invalid_col')

    def test_all_null_date_column(self):
        df = pd.DataFrame({'timestamp': [None]*5})
        with pytest.raises(ValueError):
            ProductionDataAnalyzer(df, date_col='timestamp')

# Test Data Upload with proper mocking
class TestUploadFiles:
    @patch('ProductionDataAnalyzer.files.upload')
    @patch('ProductionDataAnalyzer.zipfile.ZipFile')
    def test_colab_upload(self, mock_zip, mock_upload, tmpdir):
        mock_upload.return_value = {'test.csv': b'col1,col2\n1,2'}
        with patch('tempfile.mkdtemp', return_value=str(tmpdir)):
            df = ProductionDataAnalyzer.upload_files()
            assert len(df) == 1

    def test_local_upload(self, tmpdir):
        csv_path = tmpdir / 'test.csv'
        csv_path.write_text('col1,col2\n1,2', encoding='utf-8')
        df = ProductionDataAnalyzer.upload_files(local_files=[str(csv_path)])
        assert len(df) == 1

    def test_invalid_local_files(self):
        with pytest.raises(FileNotFoundError):
            ProductionDataAnalyzer.upload_files(local_files=['nonexistent.csv'])

# Test Aggregation with type filtering
class TestAggregation:
    def test_daily_aggregation(self, sample_data):
        analyzer = ProductionDataAnalyzer(sample_data, date_col='timestamp')
        numeric_cols = sample_data.select_dtypes(include=[np.number]).columns
        sample_data[numeric_cols] = sample_data[numeric_cols].astype('float32')
        analyzer.aggregate_data(period='day')
        assert not analyzer.agg_data.empty

    def test_invalid_period(self, sample_data):
        analyzer = ProductionDataAnalyzer(sample_data, date_col='timestamp')
        with pytest.raises(ValueError):
            analyzer.aggregate_data(period='invalid')

# Test Parameter Limits with validation mocking
class TestParameterLimits:
    def test_dict_limits(self, sample_data):
        analyzer = ProductionDataAnalyzer(sample_data, date_col='timestamp')
        limits = {'temperature': (60, 80)}
        analyzer.set_parameter_limits(limits)
        assert analyzer.param_limits == {'temperature': (60.0, 80.0)}

    @patch('ProductionDataAnalyzer.gspread')
    def test_gsheet_limits(self, mock_gspread, sample_data):
        mock_sheet = MagicMock()
        mock_gspread.authorize().open_by_url.return_value = mock_sheet
        mock_sheet.get_worksheet(0).get_all_records.return_value = [
            {'parameter': 'temperature', 'LSL': 60, 'USL': 80}
        ]
        analyzer = ProductionDataAnalyzer(sample_data, date_col='timestamp')
        analyzer.set_parameter_limits('https://docs.google.com/spreadsheets/d/test')
        assert analyzer.param_limits == {'temperature': (60.0, 80.0)}

# Test Saving with proper error handling
class TestSaving:
    def test_save_to_csv(self, sample_data):
        analyzer = ProductionDataAnalyzer(sample_data, date_col='timestamp')
        with tempfile.NamedTemporaryFile(suffix='.csv') as tmpfile:
            analyzer.save_to_csv(sample_data, filename=tmpfile.name)
            assert Path(tmpfile.name).stat().st_size > 0

    def test_save_aggregated_data(self, sample_data):
        analyzer = ProductionDataAnalyzer(sample_data, date_col='timestamp')
        analyzer.aggregate_data(period='day')
        with tempfile.NamedTemporaryFile(suffix='.csv') as tmpfile:
            analyzer.save_aggregated_data(tmpfile.name)
            assert Path(tmpfile.name).stat().st_size > 0

# Test Exceptions with better coverage
class TestExceptions:
    def test_invalid_date_conversion(self, invalid_data):
        with pytest.raises(ValueError):
            ProductionDataAnalyzer(invalid_data, date_col='timestamp')

    def test_invalid_limit_format(self, sample_data):
        analyzer = ProductionDataAnalyzer(sample_data, date_col='timestamp')
        with pytest.raises(ValueError):
            analyzer.set_parameter_limits({'temperature': [60, 80, 90]})

    def test_missing_limit_columns(self, sample_data):
        df = pd.DataFrame({'param': ['temp'], 'LSL': [60]})
        analyzer = ProductionDataAnalyzer(sample_data, date_col='timestamp')
        with pytest.raises(ValueError):
            analyzer.set_parameter_limits(df)

# Additional test for type handling in aggregation
class TestTypeHandling:
    def test_non_numeric_aggregation(self, sample_data):
        analyzer = ProductionDataAnalyzer(sample_data, date_col='timestamp')
        sample_data['category'] = 'test'
        with pytest.raises(TypeError):
            analyzer.aggregate_data(period='day')

# Mock missing modules
@pytest.fixture(autouse=True)
def mock_missing_modules(monkeypatch):
    monkeypatch.setattr('ProductionDataAnalyzer.Archive', MagicMock())
    monkeypatch.setattr('ProductionDataAnalyzer.gspread', MagicMock())