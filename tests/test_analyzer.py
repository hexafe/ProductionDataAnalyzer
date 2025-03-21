import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from ProductionDataAnalyzer import ProductionDataAnalyzer
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

# Sample data fixtures
@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='D'),
        'temperature': np.random.normal(70, 5, 100),
        'pressure': np.random.uniform(80, 120, 100),
        'serial_number': [f'SN{i:04d}' for i in range(100)]
    })

@pytest.fixture
def invalid_data():
    return pd.DataFrame({
        'timestamp': ['invalid'] * 5,
        'value': [1, 2, 3, 4, 5]
    })

# Test Initialization
class TestInitialization:
    def test_valid_initialization(self, sample_data):
        analyzer = ProductionDataAnalyzer(sample_data, date_col='timestamp')
        assert isinstance(analyzer, ProductionDataAnalyzer)
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

# Test Data Upload
class TestUploadFiles:
    @patch('ProductionDataAnalyzer.files.upload')
    @patch('ProductionDataAnalyzer.Archive.extractall')
    def test_colab_upload(self, mock_extract, mock_upload, tmpdir):
        mock_upload.return_value = {'test.csv': b'col1,col2\n1,2'}
        with patch('tempfile.mkdtemp', return_value=str(tmpdir)):
            df = ProductionDataAnalyzer.upload_files()
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 1

    def test_local_upload(self, tmpdir):
        csv_path = tmpdir / 'test.csv'
        csv_path.write_text('col1,col2\n1,2', encoding='utf-8')
        df = ProductionDataAnalyzer.upload_files(local_files=[str(csv_path)])
        assert len(df) == 1

    def test_invalid_local_files(self):
        with pytest.raises(FileNotFoundError):
            ProductionDataAnalyzer.upload_files(local_files=['nonexistent.csv'])

# Test Aggregation
class TestAggregation:
    def test_daily_aggregation(self, sample_data):
        analyzer = ProductionDataAnalyzer(sample_data, date_col='timestamp')
        analyzer.aggregate_data(period='day')
        assert not analyzer.aggregate_data.empty

    def test_invalid_period(self, sample_data):
        analyzer = ProductionDataAnalyzer(sample_data, date_col='timestamp')
        with pytest.raises(ValueError):
            analyzer.aggregate_data(period='invalid')

# Test Parameter Limits
class TestParameterLimits:
    def test_dict_limits(self, sample_data):
        analyzer = ProductionDataAnalyzer(sample_data, date_col='timestamp')
        limits = {'temperature': (60, 80)}
        analyzer.set_parameter_limits(limits)
        assert analyzer.param_limits == {'temperature': (60.0, 80.0)}

    @patch('ProductionDataAnalyzer.gspread.authorize')
    def test_gsheet_limits(self, mock_auth, sample_data):
        mock_sheet = MagicMock()
        mock_auth.return_value.open_by_url.return_value = mock_sheet
        mock_sheet.get_worksheet(0).get_all_records.return_value = [
            {'parameter': 'temperature', 'LSL': 60, 'USL': 80}
        ]
        analyzer = ProductionDataAnalyzer(sample_data, date_col='timestamp')
        analyzer.set_parameter_limits('https://docs.google.com/spreadsheets/d/test')
        assert analyzer.param_limits == {'temperature': (60.0, 80.0)}

# Test EDA Report
class TestEDAReport:
    def test_eda_report_generation(self, sample_data):
        analyzer = ProductionDataAnalyzer(sample_data, date_col='timestamp')
        report = analyzer.generate_eda_report()
        assert 'summary_stats' in report
        assert 'missing_data' in report
        assert 'distribution_plots' in report

# Test Saving
class TestSaving:
    def test_save_to_csv(self, sample_data):
        analyzer = ProductionDataAnalyzer(sample_data, date_col='timestamp')
        with tempfile.TemporaryDirectory() as tmpdirname:
            filename = Path(tmpdirname) / 'test.csv'
            analyzer.save_to_csv(sample_data, filename=str(filename), force_download=True)
            assert filename.exists()

    def test_save_aggregated_data(self, sample_data):
        analyzer = ProductionDataAnalyzer(sample_data, date_col='timestamp')
        analyzer.aggregate_data(period='day')
        with tempfile.TemporaryDirectory() as tmpdirname:
            filename = Path(tmpdirname) / 'aggregated.csv'
            analyzer.save_aggregated_data(str(filename))
            assert filename.exists()

# Test Plots
class TestPlots:
    def test_feature_distributions(self, sample_data):
        analyzer = ProductionDataAnalyzer(sample_data, date_col='timestamp')
        fig = analyzer.plot_feature_distributions()
        assert fig is not None

    def test_timeline_plot(self, sample_data):
        analyzer = ProductionDataAnalyzer(sample_data, date_col='timestamp')
        fig = analyzer.plot_interactive_timeline()
        assert fig is not None

# Test Exceptions
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