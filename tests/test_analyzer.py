# tests/test_analyzer.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock, call
from pathlib import Path
import tempfile
import shutil
import plotly.graph_objects as go
import panel as pn
from panel.widgets import MultiSelect
from ProductionDataAnalyzer.analyzer import ProductionDataAnalyzer

# Fixtures ----------------------------------------------------------------

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=100, freq='h'),
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
    # @patch('google.colab.files.upload', return_value={'test.zip': b'content'})
    # @patch('pyunpack.Archive')
    # @patch('google.colab.files.download')
    # def test_upload_process(self, mock_download, mock_archive, mock_upload, tmp_path):
    #     extracted_csv = tmp_path / 'extracted.csv'
    #     def fake_extractall(_):
    #         extracted_csv.write_text(
    #             'timestamp;temperature\n2023-01-01 00:00:00;50\n2023-01-01 01:00:00;52'
    #         )
    #     mock_archive.return_value.extractall.side_effect = fake_extractall 
    #     result = ProductionDataAnalyzer.upload_files(tmp_dir=str(tmp_path))
    #     assert not result.empty

    def test_post_merge_cleanup(self, sample_data):
        duplicated = pd.concat([sample_data, sample_data])
        cleaned = ProductionDataAnalyzer._post_merge_cleanup(
            duplicated, 'timestamp', None
        )
        assert len(cleaned) == len(sample_data)
        assert pd.api.types.is_datetime64_any_dtype(cleaned['timestamp'])

    def test_dtype_optimization(self):
        test_df = pd.DataFrame({
            'str_num': ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1'],
            'category': ['A'] * 10,
            'float': range(10)
        })
        optimized = ProductionDataAnalyzer._optimize_dtypes(test_df, None, None)
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

    @patch('google.colab.files.download')
    @patch('os.path.getsize', return_value=1024)
    @patch('pandas.DataFrame.to_csv')
    def test_save_aggregated_data(self, mock_to_csv, mock_getsize, mock_download, analyzer):
        analyzer.aggregate_data('day')
        test_path = 'test.csv'
        analyzer.save_aggregated_data(test_path)
        
        mock_to_csv.assert_called_once_with(
            test_path,
            sep=';',
            decimal=',',
            index=False,
            encoding='utf-8',
            date_format='%Y-%m-%d %H:%M:%S'
        )
        mock_getsize.assert_called_once_with(test_path)
        mock_download.assert_called_once_with(test_path)

# Parameter Limits Tests ---------------------------------------------------

class TestParameterLimits:
    def test_dict_limits(self, analyzer, limits_dict):
        analyzer.set_parameter_limits(limits_dict)
        assert analyzer.param_limits == limits_dict

    # @patch('google.colab.files.download')
    # @patch('google.auth.default')
    # @patch('gspread.authorize')
    # def test_gsheet_limits(self, mock_download, mock_auth, mock_gsheet, analyzer):
    #     # Mock Google Sheets response
    #     mock_sheet = MagicMock()
    #     mock_sheet.get_all_records.return_value = [
    #         {'parameter': 'temperature', 'LSL': 40, 'USL': 60},
    #         {'parameter': 'pressure', 'LSL': 90, 'USL': 110}
    #     ]
    #     mock_gsheet.return_value.open_by_url.return_value.get_worksheet.return_value = mock_sheet
        
    #     analyzer.set_parameter_limits(
    #         'https://docs.google.com/spreadsheets/d/test'
    #     )
    #     assert analyzer.param_limits == {
    #         'temperature': (40, 60),
    #         'pressure': (90, 110)
    #     }

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
            with pytest.raises(ValueError) as exc_info:
                ProductionDataAnalyzer.upload_files(tmp_dir=str(tmp_path), archive_ext=('.zip',))
            assert "No valid data files processed" in str(exc_info.value)

class TestConcatEdgeCases:
    @patch('ProductionDataAnalyzer.analyzer.files.upload')
    @patch('ProductionDataAnalyzer.analyzer.pd.read_csv')
    def test_mismatched_columns_preserves_rows(self, mock_read, mock_upload, tmp_path):
        mock_upload.return_value = {
            'file1.csv': b'content',
            'file2.csv': b'content'
        }
        
        df1 = pd.DataFrame({
            'timestamp': ['01.01.2023 00:00', '02.01.2023 00:00'],
            'temperature': [25, 26],
            'id': ['A', 'B']
        })
        
        df2 = pd.DataFrame({
            'timestamp': ['03.01.2023 00:00'],
            'pressure': [100],
            'id': ['C']
        })
        
        mock_read.side_effect = [df1, df2]
        
        combined = ProductionDataAnalyzer.upload_files(
            date_col='timestamp',
            tmp_dir=str(tmp_path),
            chunksize=None
        )
        
        assert len(combined) == 3
        assert set(combined.columns) == {'timestamp', 'temperature', 'pressure', 'id'}

    def test_duplicate_removal_criteria(self):
        df = pd.DataFrame({
            'timestamp': [
                '01.01.2023 00:00',
                '01.01.2023 00:00',
                '02.01.2023 00:00'
            ],
            'value': [1, 1, 2],
            'id': ['A', 'A', pd.NA]
        })

        cleaned = ProductionDataAnalyzer._post_merge_cleanup(df, 'timestamp', None)
        assert len(cleaned) == 2
        assert cleaned['id'].tolist() == ['A', pd.NA]

    def test_numeric_id_conversion(self):
        df = pd.DataFrame({
            'id': ['001', '002', 'ABC'],
            'value': [1, 2, 3]
        })
        optimized = ProductionDataAnalyzer._optimize_dtypes(df, None, None)
        assert pd.api.types.is_categorical_dtype(optimized['id']) or pd.api.types.is_string_dtype(optimized['id'])

    def test_partial_numeric_conversion(self):
        df = pd.DataFrame({
            'mixed_col': ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'X']
        })
        optimized = ProductionDataAnalyzer._optimize_dtypes(df, None, None)
        assert pd.api.types.is_string_dtype(optimized['mixed_col'])

    def test_datetime_conversion_failures(self):
        df = pd.DataFrame({
            'timestamp': [
                '01.01.2023 00:00',
                '2023-01-02 12:00',
                'invalid_date',
                '04.01.2023 18:00'
            ],
            'value': [1, 2, 3, 4]
        })

        cleaned = ProductionDataAnalyzer._post_merge_cleanup(df, 'timestamp', None)

        assert len(cleaned) == 4
        assert cleaned['timestamp'].isna().sum() == 2
    
class TestPipelineIntegration:
    @patch('ProductionDataAnalyzer.analyzer.files.upload')
    @patch('ProductionDataAnalyzer.analyzer.pd.read_csv')
    def test_full_pipeline_with_missing_columns(self, mock_read, mock_upload, tmp_path):
        mock_upload.return_value = {
            'file1.csv': b'content',
            'file2.csv': b'content'
        }
        
        df1 = pd.DataFrame({
            'timestamp': ['01.01.2023 00:00', '02.01.2023 00:00'],
            'temperature': [25, 26],
            'part_id': ['A', 'B']
        })
        
        df2 = pd.DataFrame({
            'timestamp': ['03.01.2023 00:00'],
            'pressure': [100],
            'serial_no': ['C']
        })
        
        mock_read.side_effect = [df1, df2]
        
        combined = ProductionDataAnalyzer.upload_files(
            date_col='timestamp',
            tmp_dir=str(tmp_path),
            chunksize=None
        )
        
        assert 'part_id' in combined.columns
        assert 'serial_no' in combined.columns

    def test_dtype_optimization_roundtrip(self):
        original = pd.DataFrame({
            'timestamp': ['01.01.2023 00:00']*10,
            'mixed_col': ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'X'],
            'id': [f"ID_{i}" for i in range(10)]
        })
        optimized = ProductionDataAnalyzer._optimize_dtypes(original, 'timestamp', None)

        assert pd.api.types.is_datetime64_any_dtype(optimized['timestamp'])
        assert pd.api.types.is_string_dtype(optimized['mixed_col'])
        assert pd.api.types.is_string_dtype(optimized['id']) or pd.api.types.is_categorical_dtype(optimized['id'])
        assert len(optimized) == len(original)
        assert set(optimized['id']) == set(original['id'])

# Utility Tests ------------------------------------------------------------

class TestUtilities:
    @patch('pandas.DataFrame.to_csv')
    @patch('os.path.getsize', return_value=1024)
    @patch('google.colab.files.download')
    def test_csv_save(self, mock_download, mock_getsize, mock_to_csv):
        test_df = pd.DataFrame({'test': [1, 2, 3]})
        test_path = 'test.csv'
        ProductionDataAnalyzer.save_to_csv(test_df, test_path)
        mock_to_csv.assert_called_once_with(
            test_path,
            sep=';',
            decimal=',',
            index=False,
            encoding='utf-8',
            date_format='%Y-%m-%d %H:%M:%S'
        )

    def test_invalid_csv_params(self):
        with pytest.raises(ValueError):
            ProductionDataAnalyzer.save_to_csv(
                pd.DataFrame(), 'invalid.txt'
            )

class TestEDAGeneration:
    def test_generate_eda_report_structure(self, analyzer):
        report = analyzer.generate_eda_report()
        assert isinstance(report, dict)
        assert 'summary_stats' in report
        assert 'missing_data' in report
        assert 'distribution_plots' in report
        assert 'correlations' in report
        assert 'temporal_trends' in report

    def test_eda_report_with_sampling(self, analyzer):
        report = analyzer.generate_eda_report(sample_size=50)
        assert len(report['distribution_plots'].data) > 0  # At least one trace

    def test_summary_stats_content(self, analyzer):
        stats = analyzer._get_summary_stats()
        assert not stats.empty
        assert all(col in stats.columns for col in ['mean', 'std', 'min', 'max'])

    def test_summary_stats_no_numeric(self):
        df = pd.DataFrame({'category': ['A', 'B', 'C']})
        analyzer = ProductionDataAnalyzer(df)
        with pytest.raises(ValueError):
            analyzer._get_summary_stats()

    def test_missing_data_analysis(self, analyzer):
        missing_df = analyzer._analyze_missing_data()
        assert not missing_df.empty
        assert 'missing_pct' in missing_df.columns
        assert missing_df['missing_pct'].between(0, 1).all()

    def test_missing_data_empty_input(self):
        empty_df = pd.DataFrame()
        analyzer = ProductionDataAnalyzer(empty_df)
        with pytest.raises(RuntimeError):
            analyzer._analyze_missing_data()

class TestVisualizations:
    def test_feature_distribution_plot(self, analyzer):
        fig = analyzer.plot_feature_distributions()
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # At least histogram and box plot

    def test_feature_stats_calculation(self, analyzer, limits_dict):
        analyzer.set_parameter_limits(limits_dict)
        stats = analyzer._calculate_feature_stats(analyzer.production, 'temperature')
        assert 'Defects' in stats
        assert 'Ppk' in stats

    def test_correlation_analysis(self, analyzer):
        corr_data = analyzer.analyze_correlations()
        assert 'pearson' in corr_data
        assert isinstance(corr_data['plot'], go.Figure)

    def test_correlation_insufficient_columns(self):
        df = pd.DataFrame({'col1': [1, 2, 3]})
        analyzer = ProductionDataAnalyzer(df)
        with pytest.raises(ValueError):
            analyzer.analyze_correlations()

    def test_timeline_plot(self, analyzer):
        fig = analyzer.plot_interactive_timeline(['temperature'])
        assert len(fig.data) >= 1
        assert any(trace.name == 'temperature' for trace in fig.data)

    def test_timeline_invalid_params(self, analyzer):
        with pytest.raises(ValueError):
            analyzer.plot_interactive_timeline(['invalid_param'])

    def test_timeline_missing_date_col(self):
        df = pd.DataFrame({'value': [1, 2, 3]})
        analyzer = ProductionDataAnalyzer(df)
        with pytest.raises(ValueError):
            analyzer.plot_interactive_timeline()

class TestDashboard:
    @patch('panel.Column')
    @patch('panel.widgets.MultiSelect')
    def test_dashboard_creation(self, mock_select, mock_col, analyzer):
        mock_select.return_value = MultiSelect(
            options=['temperature'], 
            value=['temperature']
        )
        mock_col.return_value = pn.Column("Test Content")
        
        dashboard = analyzer.create_interactive_dashboard()
        assert isinstance(dashboard, pn.Column)

    def test_dashboard_environment_error(self):
        with patch('panel.extension', side_effect=ImportError):
            with pytest.raises(RuntimeError):
                analyzer = ProductionDataAnalyzer(pd.DataFrame())
                analyzer.create_interactive_dashboard()

class TestDataFrameProcessing:
    def test_dataframe_limit_processing(self, analyzer):
        limits_df = pd.DataFrame({
            'parameter': ['temperature', 'pressure'],
            'LSL': [40, 90],
            'USL': [60, 110]
        })
        result = analyzer._process_dataframe_source(limits_df)
        assert result == {'temperature': (40, 60), 'pressure': (90, 110)}

    def test_invalid_dataframe_columns(self, analyzer):
        invalid_df = pd.DataFrame({'wrong_col': [1, 2]})
        with pytest.raises(ValueError):
            analyzer._process_dataframe_source(invalid_df)

    def test_duplicate_parameters(self, analyzer):
        dup_df = pd.DataFrame({
            'parameter': ['temp', 'temp'],
            'LSL': [10, 20],
            'USL': [30, 40]
        })
        with pytest.raises(ValueError):
            analyzer._process_dataframe_source(dup_df)

    def test_non_numeric_limits(self, analyzer):
        str_df = pd.DataFrame({
            'parameter': ['temp'],
            'LSL': ['low'],
            'USL': ['high']
        })
        with pytest.raises(TypeError):
            analyzer._process_dataframe_source(str_df)

# Main ---------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main(["-v", "--cov=ProductionDataAnalyzer", "--cov-report=html"])
