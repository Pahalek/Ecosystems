# Econometric Agent

A professional econometric modeling system that fetches real economic data from validated sources, performs comprehensive data quality validation, and builds sophisticated econometric models for economic analysis and forecasting.

## Features

- **Real Economic Data Access**: Fetch data from FRED (Federal Reserve Economic Data), World Bank, and other reliable sources
- **Data Quality Validation**: Comprehensive data quality assessment with missing value detection, outlier analysis, and temporal coverage validation
- **Professional Econometric Models**: Linear regression, regularized regression (Ridge, Lasso, ElasticNet), ARIMA, SARIMA, and Vector Autoregression (VAR)
- **Stationarity Testing**: Augmented Dickey-Fuller (ADF) and KPSS tests for time series analysis
- **Automated Model Comparison**: Compare multiple models with standardized performance metrics
- **Forecasting Capabilities**: Generate forecasts with confidence intervals
- **Comprehensive Reporting**: Detailed analysis reports with model diagnostics and recommendations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Pahalek/Ecosystems.git
cd Ecosystems
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Set up FRED API access:
   - Get a free API key from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html)
   - Set environment variable: `export FRED_API_KEY=your_api_key_here`

## Quick Start

```python
from econometric_agent import EconometricAgent

# Initialize the agent
agent = EconometricAgent()

# Load economic indicators
data = agent.load_economic_indicators(
    start_date='2020-01-01',
    end_date='2023-12-31'
)

# Validate data quality
validation_results = agent.validate_data()

# Clean the data
cleaned_data = agent.clean_data(interpolate_missing=True)

# Build a regression model
results = agent.build_regression_model(
    target_variable='GDP',
    feature_variables=['UNEMPLOYMENT', 'INFLATION']
)

# Build a time series model
arima_results = agent.build_time_series_model(
    target_variable='GDP',
    model_type='arima'
)

# Generate forecasts
forecast = agent.generate_forecast(
    model_name=arima_results['model_name'],
    steps=12
)

# Generate comprehensive report
report = agent.generate_report()
print(report)
```

## Detailed Usage

### 1. Data Fetching

#### FRED Data
```python
# Get specific indicators
data = agent.load_economic_indicators(
    indicators=['GDP', 'UNRATE', 'CPIAUCSL'],
    start_date='2020-01-01'
)

# Get common economic indicators
data = agent.load_economic_indicators()

# Search for indicators
results = agent.list_available_indicators('unemployment')
```

#### Custom Data
```python
# Load your own data
import pandas as pd
custom_data = pd.read_csv('your_data.csv', index_col=0, parse_dates=True)
agent.datasets['custom'] = custom_data
```

### 2. Data Quality Analysis

```python
# Comprehensive validation
validation = agent.validate_data('main')

# Generate quality report
report = agent.data_validator.generate_quality_report(data)
print(report)

# Clean data with options
cleaned = agent.clean_data(
    remove_outliers=True,
    interpolate_missing=True,
    method='linear'
)
```

### 3. Econometric Modeling

#### Linear Regression
```python
results = agent.build_regression_model(
    target_variable='GDP',
    feature_variables=['UNEMPLOYMENT', 'INFLATION'],
    test_size=0.2
)
```

#### Regularized Regression
```python
# Ridge regression
ridge_results = agent.build_regression_model(
    target_variable='GDP',
    model_type='ridge',
    alpha=1.0
)

# Lasso regression  
lasso_results = agent.build_regression_model(
    target_variable='GDP',
    model_type='lasso',
    alpha=0.1
)
```

#### Time Series Models
```python
# ARIMA model
arima_results = agent.build_time_series_model(
    target_variable='GDP',
    model_type='arima',
    order=(1, 1, 1)  # Optional: auto-selected if not provided
)

# SARIMA model
sarima_results = agent.build_time_series_model(
    target_variable='GDP',
    model_type='arima',
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12)
)

# Vector Autoregression
var_results = agent.build_time_series_model(
    target_variable=['GDP', 'UNEMPLOYMENT', 'INFLATION'],
    model_type='var',
    maxlags=5
)
```

### 4. Model Analysis

#### Stationarity Testing
```python
stationarity = agent.analyze_stationarity('GDP')
print(f"Series is {stationarity['interpretation']}")
```

#### Model Comparison
```python
comparison = agent.compare_models()
print(comparison)
```

#### Forecasting
```python
# Generate forecasts
forecast = agent.generate_forecast('arima_GDP_(1, 1, 1)', steps=12)

# Forecasts include confidence intervals for ARIMA models
print(forecast[['forecast', 'lower_ci', 'upper_ci']])
```

## Available Economic Indicators

When using FRED data, common indicators include:

- **GDP**: Gross Domestic Product
- **UNRATE**: Unemployment Rate  
- **CPIAUCSL**: Consumer Price Index (Inflation)
- **FEDFUNDS**: Federal Funds Rate
- **INDPRO**: Industrial Production Index
- **RSAFS**: Retail Sales
- **UMCSENT**: Consumer Sentiment
- **SP500**: S&P 500 Index

## Model Types

### Regression Models
- **Linear Regression**: OLS with comprehensive statistics
- **Ridge Regression**: L2 regularization for multicollinearity
- **Lasso Regression**: L1 regularization with feature selection
- **ElasticNet**: Combined L1/L2 regularization

### Time Series Models  
- **ARIMA**: Autoregressive Integrated Moving Average
- **SARIMA**: Seasonal ARIMA for seasonal data
- **VAR**: Vector Autoregression for multivariate analysis

## Data Quality Features

- **Missing Data Analysis**: Detection and handling of missing values
- **Outlier Detection**: Multiple methods (IQR, Z-score, Modified Z-score)
- **Temporal Coverage**: Gap detection and frequency analysis
- **Data Type Validation**: Ensuring appropriate data types
- **Quality Scoring**: Overall data quality assessment (0-100)

## Example Output

### Data Quality Report
```
============================================================
DATA QUALITY REPORT
============================================================
Total Observations: 48
Quality Score: 85.0/100

MISSING DATA ANALYSIS:
------------------------------
✓ GDP: 0 missing (0.0%)
✓ UNEMPLOYMENT: 0 missing (0.0%)
✓ INFLATION: 2 missing (4.2%)

OUTLIER ANALYSIS:
------------------------------
✓ GDP: 0 outliers (0.0%)
⚠ UNEMPLOYMENT: 3 outliers (6.2%)
✓ INFLATION: 1 outliers (2.1%)

RECOMMENDATIONS:
------------------------------
1. Column 'UNEMPLOYMENT' has 6.2% outliers. Review for data errors.
```

### Model Results
```
✓ Linear regression model built successfully
  R² (Test): 0.8234
  MSE (Test): 2.4567

Model Comparison:
==================================================
Model                    R²_Test    MSE_Test    AIC
linear_regression_GDP    0.8234     2.4567      -
ridge_GDP_alpha1.0       0.8156     2.5234      -
arima_GDP_(2,1,1)        -          -           145.23
```

## Testing

Run the test suite:
```bash
python tests/test_econometric_agent.py
```

Run the example:
```bash
python example.py
```

## Requirements

- Python 3.7+
- pandas >= 2.0.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- statsmodels >= 0.14.0
- scikit-learn >= 1.3.0
- fredapi >= 0.5.0 (for FRED data access)
- requests >= 2.31.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0

## API Reference

### EconometricAgent

Main class for econometric analysis workflow.

#### Methods

- `load_economic_indicators()`: Load data from FRED or World Bank
- `validate_data()`: Perform comprehensive data quality analysis  
- `clean_data()`: Clean and preprocess data
- `build_regression_model()`: Build linear or regularized regression models
- `build_time_series_model()`: Build ARIMA, SARIMA, or VAR models
- `analyze_stationarity()`: Test time series stationarity
- `generate_forecast()`: Generate model forecasts
- `compare_models()`: Compare multiple model performances
- `generate_report()`: Create comprehensive analysis reports

### EconomicDataFetcher

Handles data retrieval from economic data sources.

#### Methods

- `get_fred_data()`: Fetch single FRED series
- `get_multiple_fred_series()`: Fetch multiple FRED series
- `get_world_bank_data()`: Fetch World Bank indicators
- `get_common_economic_indicators()`: Fetch standard economic indicators
- `search_fred_series()`: Search FRED database

### DataValidator  

Validates and cleans economic data.

#### Methods

- `validate_data_quality()`: Comprehensive quality assessment
- `clean_data()`: Clean data based on validation results
- `generate_quality_report()`: Generate formatted quality report

### EconometricModels

Builds and manages econometric models.

#### Methods

- `linear_regression()`: OLS regression with full statistics
- `regularized_regression()`: Ridge, Lasso, ElasticNet regression
- `time_series_arima()`: ARIMA/SARIMA models
- `vector_autoregression()`: VAR models for multivariate analysis
- `test_stationarity()`: ADF and KPSS stationarity tests
- `model_comparison()`: Compare model performances

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Federal Reserve Bank of St. Louis for FRED data access
- World Bank for economic indicators
- The statsmodels and scikit-learn communities for excellent econometric tools
