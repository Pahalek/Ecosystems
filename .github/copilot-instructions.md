# Econometric Agent - GitHub Copilot Instructions

**ALWAYS follow these instructions first and only search or use bash commands when you encounter unexpected information that does not match the info here.**

## Working Effectively

### Bootstrap and Dependencies
Install all required dependencies:
```bash
cd /path/to/Ecosystems
pip install -r requirements.txt  # Takes 32 seconds. NEVER CANCEL. Set timeout to 120+ seconds.
```

Additional dependencies for Amazon analysis (optional):
```bash
pip install yfinance  # Takes 24 seconds. NEVER CANCEL. Set timeout to 120+ seconds.
```

### Test Installation
Verify the installation works:
```bash
python -c "from econometric_agent import EconometricAgent; print('Import successful')"  # Takes 4 seconds
```

### Run Tests
Execute the test suite:
```bash
python tests/test_econometric_agent.py  # Takes 2 seconds. Always run to verify functionality.
```
Expected output: "9 tests... OK" with all tests passing.

### Run Examples
Basic econometric functionality:
```bash
python example.py  # Takes 3 seconds. Demonstrates core features with synthetic data.
```

Advanced professional workflow:
```bash
python advanced_example.py  # Takes 4 seconds. Shows comprehensive analysis workflow.
```

### API Configuration (Optional)
For real economic data access, configure API keys:
1. Copy environment template: `cp .env.example .env`
2. Edit `.env` and add your FRED API key: `FRED_API_KEY=your_key_here`
3. Test configuration: `python test_fred_api.py` (Takes <1 second)

Without API keys, all examples work with synthetic data that closely mimics real economic indicators.

## Project Structure

### Core Package: `/econometric_agent/`
- `agent.py` - Main EconometricAgent class with full workflow
- `data_fetcher.py` - Economic data fetching (FRED, synthetic fallback)
- `data_validator.py` - Data quality validation and cleaning
- `models.py` - Econometric modeling (ARIMA, VAR, regression)

### Example Scripts
- `example.py` - Basic demo (3 seconds runtime)
- `advanced_example.py` - Professional workflow (4 seconds runtime)
- `amazon_econometric_analysis.py` - Amazon ecosystem analysis (requires yfinance + internet)
- `professional_analysis.py` - FRED economic analysis (requires FRED API key)

### Tests: `/tests/`
- `test_econometric_agent.py` - Comprehensive test suite (2 seconds runtime)

## Validation Scenarios

**ALWAYS run these validation steps after making changes:**

1. **Import Test**: Verify package imports correctly
   ```bash
   python -c "from econometric_agent import EconometricAgent; print('Success')"
   ```

2. **Basic Functionality**: Run core workflow
   ```bash
   python example.py
   ```
   Expected: Completes in 3 seconds with synthetic data, builds models, generates forecasts.

3. **Advanced Features**: Test professional workflow
   ```bash
   python advanced_example.py
   ```
   Expected: Completes in 4 seconds with multiple models, stationarity analysis, 12-month forecasts.

4. **Test Suite**: Verify all tests pass
   ```bash
   python tests/test_econometric_agent.py
   ```
   Expected: 9 tests pass in under 2 seconds.

**Manual Validation Requirements:**
- Verify console output shows data loading, model building, and forecast generation
- Check that R² scores, AIC/BIC values, and forecasts are reasonable
- Confirm quality scores are between 0-100
- Ensure no exceptions or errors in normal operation

## Network and API Dependencies

### Internet Access Required
- `amazon_econometric_analysis.py` - Fetches Amazon stock data via yfinance
- Real FRED data via `fredapi` (falls back to synthetic data gracefully)

### Offline Functionality
- Core econometric_agent package works entirely offline
- All examples work with synthetic data when APIs unavailable
- Test suite runs completely offline

### API Key Setup
- **FRED API**: Free from https://fred.stlouisfed.org/docs/api/api_key.html
- **Alpha Vantage**: Optional, for enhanced financial data
- **Quandl**: Optional, for additional datasets

## Common Operations

### Data Loading and Analysis
```python
from econometric_agent import EconometricAgent

# Initialize (works offline with synthetic data)
agent = EconometricAgent()

# Load data (real if API key available, synthetic otherwise)
data = agent.load_economic_indicators(start_date='2020-01-01')

# Validate data quality
validation = agent.validate_data()  # Returns quality score 0-100

# Clean data
cleaned = agent.clean_data(interpolate_missing=True)
```

### Model Building
```python
# Linear regression
regression_results = agent.build_regression_model(
    target_variable='GDP',
    feature_variables=['UNEMPLOYMENT', 'INFLATION']
)

# Time series modeling
arima_results = agent.build_time_series_model(
    target_variable='GDP',
    model_type='arima'
)

# Generate forecasts
forecast = agent.generate_forecast(
    model_name=arima_results['model_name'],
    steps=12  # 12-month forecast
)
```

## Timing Expectations

**NEVER CANCEL any operation. All core operations complete quickly:**

- **Installation**: 32 seconds for core deps, +24 seconds for yfinance
- **Import/Setup**: 4 seconds
- **Tests**: 2 seconds (9 tests)
- **Basic Example**: 3 seconds
- **Advanced Example**: 4 seconds  
- **API Test**: <1 second

**No long-running builds or operations exist in this codebase.**

## Error Handling

### Expected Behaviors
- **No FRED API key**: Scripts use synthetic data and continue normally
- **No internet**: Amazon analysis fails gracefully, core functionality unaffected
- **Missing yfinance**: Amazon analysis fails with clear ModuleNotFoundError
- **Data quality issues**: Automatically handled with data cleaning and validation

### Troubleshooting
- **Import errors**: Run `pip install -r requirements.txt`
- **FRED errors**: Check API key in `.env` file or use synthetic data mode
- **Amazon analysis errors**: Install yfinance and verify internet access
- **Test failures**: Usually indicate missing dependencies

## Development Workflow

### Making Changes
1. Run existing tests: `python tests/test_econometric_agent.py`
2. Make your changes to the econometric_agent package
3. Test core functionality: `python example.py`
4. Test advanced features: `python advanced_example.py`
5. Re-run tests to ensure no regressions

### Adding Features
- Follow existing patterns in `econometric_agent/` modules
- Add tests to `tests/test_econometric_agent.py`
- Update examples if adding major features
- Ensure graceful fallback when external APIs unavailable

### Validation Checklist
- [ ] All imports work correctly
- [ ] Basic example runs in ~3 seconds
- [ ] Advanced example runs in ~4 seconds  
- [ ] Test suite passes (9 tests in ~2 seconds)
- [ ] No exceptions in normal operation
- [ ] Reasonable output values (R² between -1 and 1, quality scores 0-100)

This is a mature, well-tested econometric analysis system optimized for both educational use (synthetic data) and professional analysis (real economic data when APIs available).