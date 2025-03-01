# Production Data Analyzer

Production data analysis toolkit with integrated quality analytics

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hexafe/ProductionDataAnalyzer)

## Features

### Data Management
- Multi-format ingestion (CSV/Excel/ZIP/7z/RAR)
- Time-indexed dataset organization
- Part ID-based filtering and segmentation
- Google Sheets integration with OAuth

### Temporal Analysis (Upcoming)
- Flexible aggregation (minute/hourly/daily/weekly/monthly)
- Statistical timeframe comparison (T-test)
- Interactive time-series visualization

### Production Statistics (Upcoming)
- Statistical Process Control (SPC) charting
- Process capability analysis (Cp/Cpk/Ppk)
- Parameter limit configuration from multiple sources
- Multivariate correlation analysis

### Machine Learning (Upcoming)
- Defect prediction models (RandomForest/XGBoost/LightGBM/CatBoost)
- Explainable AI with SHAP values
- Concept drift detection
- Automated hyperparameter tuning

## Installation

### Google Colab
```python
!git clone https://github.com/hexafe/ProductionDataAnalyzer.git
%cd ProductionDataAnalyzer
!pip install -r requirements.txt
from ProductionDataAnalyzer import ProductionDataAnalyzer
%cd ..
```

## Basic workflow
### Upload and read data from local CSV/Excel (or zip CSV/Excel) file(s)
```python
df = ProductionDataAnalyzer.upload_files(date_col='timestamp')
```

### Initialize with production data
```python
analyzer = ProductionDataAnalyzer(
    production_data=df,
    date_col='timestamp'
)
```

### Configure quality limits
```python
analyzer.set_parameter_limits({
    'pressure': (80, 120),
    'temperature': (20, 45)
})
```

### Temporal aggregation
```python
analyzer.aggregate_data(period='week')
```

### Save results
```python
analyzer.save_aggregated_data('weekly_report.csv')
```

## Multi-source Limit Configuration
### From Google Sheets
```python
analyzer.set_parameter_limits(
    source='https://docs.google.com/spreadsheets/d/your-sheet-id'
)
```

### From uploaded CSV
```python
uploaded_csv = ProductionDataAnalyzer.upload_files()
analyzer.set_parameter_limits(source=uploaded_csv)
```

### From dictionary
```python
limits_dict = {'fitting_force': (1000, 2200), 'fitting_height': (20, 20.5)}
analyzer.set_parameter_limits(source=limits_dict)
```

## Data filtering by id list in CSV/Excel file (Google Sheets and list to be added)
```python
data_df = ProductionDataAnalyzer.upload_files()
ids_df = ProductionDataAnalyzer.upload_files()
filtered_data = ProductionDataAnalyzer.filter_by_id(
    production_data_df=data_df,
    id_data_df=ids_df,
    id_col='serial_number'
)
ProductionDataAnalyzer.save_to_csv(df=filtered_data, filename="filtered_data.csv")
```

## Core methods
```upload_files()```: Multi-format data ingestion

```filter_by_id()```: Part number-based filtering

```aggregate_data()```: Temporal data aggregation

```set_parameter_limits()```: Quality limit configuration

```save_to_csv()```: Data export with formatting
