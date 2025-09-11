"""
Example usage of the Econometric Agent.

This script demonstrates how to use the Econometric Agent for:
1. Loading economic data
2. Validating data quality  
3. Building econometric models
4. Generating forecasts and reports
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from econometric_agent import EconometricAgent
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def main():
    """
    Demonstrate the Econometric Agent capabilities.
    """
    print("=" * 60)
    print("ECONOMETRIC AGENT DEMONSTRATION")
    print("=" * 60)
    
    # Initialize the agent
    # Note: For full functionality, set FRED_API_KEY environment variable
    agent = EconometricAgent()
    
    print("\n1. LOADING ECONOMIC DATA")
    print("-" * 30)
    
    try:
        # Try to load real data if FRED API key is available
        data = agent.load_economic_indicators(
            start_date='2020-01-01',
            end_date='2023-12-31'
        )
        print(f"Loaded data shape: {data.shape}")
        print(f"Variables: {list(data.columns)}")
        
    except Exception as e:
        print(f"Could not load real data (FRED API key needed): {e}")
        print("Generating synthetic data for demonstration...")
        
        # Generate synthetic economic data for demonstration
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='ME')
        np.random.seed(42)
        
        # Create synthetic economic indicators
        data = pd.DataFrame({
            'GDP': 100 + np.cumsum(np.random.normal(0.3, 2, len(dates))),
            'UNEMPLOYMENT': 5 + np.random.normal(0, 1.5, len(dates)),
            'INFLATION': 2 + np.random.normal(0, 0.8, len(dates)),
            'INTEREST_RATE': 2 + np.random.normal(0, 0.5, len(dates))
        }, index=dates)
        
        # Make unemployment and interest rates more realistic
        data['UNEMPLOYMENT'] = np.clip(data['UNEMPLOYMENT'], 2, 15)
        data['INTEREST_RATE'] = np.clip(data['INTEREST_RATE'], 0, 8)
        
        # Store synthetic data in agent
        agent.datasets['main'] = data
        agent.current_data = data
        
        print(f"Generated synthetic data shape: {data.shape}")
        print(f"Variables: {list(data.columns)}")
    
    print("\n2. DATA QUALITY VALIDATION")
    print("-" * 30)
    
    # Validate data quality
    validation_results = agent.validate_data(dataset_name='main')
    
    # Display data quality report
    quality_report = agent.data_validator.generate_quality_report(agent.current_data)
    print(quality_report)
    
    print("\n3. DATA CLEANING")
    print("-" * 30)
    
    # Clean the data
    cleaned_data = agent.clean_data(
        dataset_name='main',
        interpolate_missing=True,
        remove_outliers=False
    )
    
    print("\n4. STATIONARITY ANALYSIS")
    print("-" * 30)
    
    # Test stationarity for GDP
    stationarity_results = agent.analyze_stationarity('GDP', 'main_cleaned')
    
    print("\n5. BUILDING ECONOMETRIC MODELS")
    print("-" * 30)
    
    # Build a linear regression model
    regression_results = agent.build_regression_model(
        target_variable='GDP',
        feature_variables=['UNEMPLOYMENT', 'INFLATION', 'INTEREST_RATE'],
        dataset_name='main_cleaned'
    )
    
    # Build an ARIMA time series model
    try:
        arima_results = agent.build_time_series_model(
            target_variable='GDP',
            dataset_name='main_cleaned',
            model_type='arima'
        )
    except Exception as e:
        print(f"ARIMA model failed: {e}")
        arima_results = None
    
    print("\n6. MODEL COMPARISON")
    print("-" * 30)
    
    # Compare models
    comparison = agent.compare_models()
    
    print("\n7. FORECASTING")
    print("-" * 30)
    
    # Generate forecasts if ARIMA model was successful
    if arima_results:
        try:
            forecast = agent.generate_forecast(
                model_name=arima_results['model_name'],
                steps=6
            )
            print("6-month GDP forecast:")
            print(forecast)
        except Exception as e:
            print(f"Forecasting failed: {e}")
    
    print("\n8. COMPREHENSIVE REPORT")
    print("-" * 30)
    
    # Generate comprehensive report
    report = agent.generate_report()
    print(report)
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETED")
    print("=" * 60)
    
    # Show available datasets
    print(f"\nAvailable datasets: {list(agent.datasets.keys())}")
    print(f"Analysis history entries: {len(agent.analysis_history)}")


if __name__ == "__main__":
    # Add numpy import for synthetic data generation
    import numpy as np
    main()