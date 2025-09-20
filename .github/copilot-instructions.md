# Econometric Agent System

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

Bootstrap, build, and test the repository:
- `git clone https://github.com/Pahalek/Ecosystems.git && cd Ecosystems`
- `pip install -r requirements.txt` -- takes 35 seconds. NEVER CANCEL. Set timeout to 60+ seconds.
- Test installation: `python -c "from econometric_agent import EconometricAgent; print('✅ Install successful')"` -- takes 2 seconds.
- Run test suite: `python tests/test_econometric_agent.py` -- takes 2 seconds. NEVER CANCEL. Set timeout to 30+ seconds.

Run the system:
- ALWAYS run the installation steps first if working in a fresh environment.
- Basic demo: `python example.py` -- takes 3 seconds. Demonstrates core functionality with synthetic data.
- Advanced demo: `python advanced_example.py` -- takes 4 seconds. Comprehensive professional workflow.
- Unit tests: `python tests/test_econometric_agent.py` -- takes 2 seconds. 9 tests should pass.

## Validation

- ALWAYS manually validate any new code by running both `python example.py` and `python advanced_example.py` after making changes.
- ALWAYS run through at least one complete end-to-end scenario after making changes.
- ALWAYS run the test suite `python tests/test_econometric_agent.py` before committing changes.
- Test agent creation manually: `python -c "from econometric_agent import EconometricAgent; agent = EconometricAgent(); print('✅ Agent works')"` -- takes 2 seconds.
- There are no linting tools configured (no flake8, black, or pylint). Code style should follow existing patterns.

## FRED API Configuration (Optional but Recommended)

The system works with synthetic data by default but provides much better results with real economic data from FRED:
- Get a free API key from https://fred.stlouisfed.org/docs/api/api_key.html
- `cp .env.example .env`
- Edit `.env` and uncomment/update: `FRED_API_KEY=your_actual_api_key_here`
- Test configuration works by running examples - they will show "✓ Loaded real economic data" instead of synthetic data
- The `.env` file is gitignored for security

## Architecture

Core package structure:
```
econometric_agent/
├── __init__.py          # Main exports
├── agent.py             # EconometricAgent - main orchestrator  
├── data_fetcher.py      # EconomicDataFetcher - FRED/World Bank data
├── data_validator.py    # DataValidator - quality validation
└── models.py           # EconometricModels - regression, ARIMA, VAR
```

Key files:
- `example.py` - Basic demonstration script (3 seconds runtime)
- `advanced_example.py` - Professional workflow demo (4 seconds runtime) 
- `tests/test_econometric_agent.py` - Comprehensive test suite (2 seconds runtime)
- `requirements.txt` - Python dependencies (35 seconds install time)

## Common Tasks

### Running Examples
```bash
# Basic functionality demo
python example.py

# Advanced professional workflow  
python advanced_example.py
```

### Testing Changes
```bash
# Run all tests
python tests/test_econometric_agent.py

# Quick import test
python -c "from econometric_agent import EconometricAgent; print('OK')"
```

### Working with the Agent
```python
from econometric_agent import EconometricAgent

# Initialize agent (auto-detects FRED API key from .env)
agent = EconometricAgent()

# Load data (real data if FRED key configured, synthetic otherwise)
data = agent.load_economic_indicators(start_date='2020-01-01')

# Validate data quality  
validation = agent.validate_data('main')

# Build models
agent.build_regression_model('GDP', ['UNEMPLOYMENT', 'INFLATION'])
agent.build_time_series_model('GDP', model_type='arima')

# Generate forecasts
forecast = agent.generate_forecast('arima_GDP_*', steps=12)
```

## Expected Command Outputs

### Installation Success
```bash
$ pip install -r requirements.txt
Successfully installed [package list]

$ python -c "from econometric_agent import EconometricAgent; print('✅ Install successful')"
✅ Install successful
```

### Test Suite Success  
```bash
$ python tests/test_econometric_agent.py
...........
Ran 9 tests in 0.073s
OK
Tests PASSED
```

### Example Script Success
```bash
$ python example.py
============================================================
ECONOMETRIC AGENT DEMONSTRATION
============================================================
[... detailed output showing data loading, validation, modeling ...]
DEMONSTRATION COMPLETED
```

## Troubleshooting

**Import errors**: Run `pip install -r requirements.txt` to install dependencies

**FRED API errors**: Either add a real API key to `.env` or accept that synthetic data will be used

**Test failures**: Tests should always pass on a clean install - check if dependencies are properly installed

**Performance**: All operations should complete quickly (under 5 seconds each). If commands hang, check for dependency issues.

## File Locations

Repository root: `/home/runner/work/Ecosystems/Ecosystems`
Main package: `./econometric_agent/`
Examples: `./example.py`, `./advanced_example.py` 
Tests: `./tests/test_econometric_agent.py`
Dependencies: `./requirements.txt`
Config: `./.env` (create from `.env.example`)