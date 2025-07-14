"""
Advanced Econometric Agent Example - Professional Economic Analysis Workflow

This example demonstrates advanced features of the Econometric Agent:
1. Loading specific economic indicators
2. Advanced data quality analysis
3. Multiple model types and comparison
4. Forecasting with confidence intervals
5. Professional reporting

Run this with a FRED API key for real data analysis.
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from econometric_agent import EconometricAgent


def run_professional_analysis():
    """
    Demonstrate professional econometric analysis workflow.
    """
    print("=" * 80)
    print("PROFESSIONAL ECONOMETRIC ANALYSIS WORKFLOW")
    print("=" * 80)
    
    # Initialize agent
    fred_api_key = os.getenv('FRED_API_KEY')
    agent = EconometricAgent(fred_api_key=fred_api_key)
    
    print(f"\nFRED API Status: {'✓ Connected' if fred_api_key else '✗ No API Key (using synthetic data)'}")
    
    # Step 1: Load comprehensive economic dataset
    print("\n" + "="*60)
    print("STEP 1: LOADING ECONOMIC DATA")
    print("="*60)
    
    indicators = ['GDP', 'UNRATE', 'CPIAUCSL', 'FEDFUNDS', 'INDPRO', 'RSAFS']
    
    try:
        if fred_api_key:
            data = agent.load_economic_indicators(
                indicators=indicators,
                start_date='2010-01-01',
                end_date='2023-12-31'
            )
            print(f"✓ Loaded real economic data: {data.shape}")
        else:
            # Create realistic synthetic data for demonstration
            dates = pd.date_range('2010-01-01', '2023-12-31', freq='MS')
            np.random.seed(42)
            
            # Generate correlated economic variables
            n_obs = len(dates)
            
            # Base economic cycle
            cycle = np.sin(np.arange(n_obs) * 2 * np.pi / 12) * 2  # Annual cycle
            trend = np.linspace(0, 20, n_obs)  # Long-term growth
            
            data = pd.DataFrame({
                'GDP': 100 + trend + cycle + np.cumsum(np.random.normal(0, 0.5, n_obs)),
                'UNEMPLOYMENT': np.clip(6 - cycle/2 + np.random.normal(0, 0.8, n_obs), 2, 15),
                'INFLATION': 2 + cycle/4 + np.random.normal(0, 0.6, n_obs),
                'INTEREST_RATE': np.clip(2 + cycle/3 + np.random.normal(0, 0.4, n_obs), 0, 8),
                'INDUSTRIAL_PROD': 100 + trend*0.8 + cycle*1.5 + np.cumsum(np.random.normal(0, 0.7, n_obs)),
                'RETAIL_SALES': 100 + trend*1.2 + cycle*2 + np.cumsum(np.random.normal(0, 0.6, n_obs))
            }, index=dates)
            
            agent.datasets['main'] = data
            agent.current_data = data
            print(f"✓ Generated realistic synthetic data: {data.shape}")
    
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return
    
    # Display basic statistics
    print(f"\nDataset Overview:")
    print(f"Period: {data.index.min().strftime('%Y-%m')} to {data.index.max().strftime('%Y-%m')}")
    print(f"Variables: {', '.join(data.columns)}")
    print(f"\nDescriptive Statistics:")
    print(data.describe().round(2))
    
    # Step 2: Comprehensive data quality analysis
    print("\n" + "="*60)
    print("STEP 2: DATA QUALITY ANALYSIS")
    print("="*60)
    
    validation = agent.validate_data('main')
    quality_report = agent.data_validator.generate_quality_report(data)
    print(quality_report)
    
    # Clean data if needed
    if validation['quality_score'] < 85:
        print("\n📋 Cleaning data due to quality issues...")
        cleaned_data = agent.clean_data('main', interpolate_missing=True, remove_outliers=False)
    else:
        print("\n✓ Data quality is excellent - minimal cleaning needed")
        cleaned_data = agent.clean_data('main', interpolate_missing=True)
    
    # Step 3: Stationarity analysis for time series variables
    print("\n" + "="*60)
    print("STEP 3: STATIONARITY ANALYSIS")
    print("="*60)
    
    key_variables = ['GDP', 'UNEMPLOYMENT', 'INFLATION']
    stationarity_results = {}
    
    for var in key_variables:
        if var in data.columns:
            result = agent.analyze_stationarity(var, 'main_cleaned')
            stationarity_results[var] = result
            print(f"\n{var}:")
            print(f"  {result['interpretation']}")
            print(f"  ADF p-value: {result['adf_test']['p_value']:.4f}")
            print(f"  KPSS p-value: {result['kpss_test']['p_value']:.4f}")
    
    # Step 4: Build multiple econometric models
    print("\n" + "="*60)
    print("STEP 4: ECONOMETRIC MODELING")
    print("="*60)
    
    models_built = []
    
    # 4a. Linear Regression Models
    print("\n📊 Building Regression Models...")
    
    if 'GDP' in data.columns and 'UNEMPLOYMENT' in data.columns:
        # GDP model with multiple predictors
        features = [col for col in data.columns if col != 'GDP'][:3]  # Use first 3 non-GDP columns
        linear_result = agent.build_regression_model(
            target_variable='GDP',
            feature_variables=features,
            dataset_name='main_cleaned'
        )
        models_built.append(linear_result['model_name'])
        
        # Ridge regression for comparison
        ridge_result = agent.build_regression_model(
            target_variable='GDP',
            feature_variables=features,
            model_type='ridge',
            alpha=1.0,
            dataset_name='main_cleaned'
        )
        models_built.append(ridge_result['model_name'])
    
    # 4b. Time Series Models
    print("\n📈 Building Time Series Models...")
    
    if 'GDP' in data.columns:
        # ARIMA model for GDP
        try:
            arima_result = agent.build_time_series_model(
                target_variable='GDP',
                model_type='arima',
                dataset_name='main_cleaned'
            )
            models_built.append(arima_result['model_name'])
        except Exception as e:
            print(f"⚠ ARIMA model failed: {e}")
    
    # VAR model for multivariate analysis
    if len(data.columns) >= 3:
        try:
            var_variables = list(data.columns)[:3]  # Use first 3 variables
            var_result = agent.build_time_series_model(
                target_variable=var_variables,
                model_type='var',
                dataset_name='main_cleaned'
            )
            models_built.append(var_result['model_name'])
        except Exception as e:
            print(f"⚠ VAR model failed: {e}")
    
    # Step 5: Model comparison and evaluation
    print("\n" + "="*60)
    print("STEP 5: MODEL COMPARISON")
    print("="*60)
    
    if models_built:
        comparison = agent.compare_models(models_built)
        print("\nModel Performance Comparison:")
        print("="*50)
        print(comparison.to_string(index=False, float_format='%.4f'))
        
        # Identify best model
        if 'R²_Test' in comparison.columns:
            best_r2_idx = comparison['R²_Test'].idxmax()
            best_model = comparison.loc[best_r2_idx, 'Model']
            print(f"\n🏆 Best R² Performance: {best_model} (R² = {comparison.loc[best_r2_idx, 'R²_Test']:.4f})")
    
    # Step 6: Generate forecasts
    print("\n" + "="*60)
    print("STEP 6: FORECASTING")
    print("="*60)
    
    forecasts_generated = []
    
    for model_name in models_built:
        if 'arima' in model_name.lower():
            try:
                forecast = agent.generate_forecast(model_name, steps=12)
                forecasts_generated.append((model_name, forecast))
                
                print(f"\n📊 12-Month Forecast using {model_name}:")
                if 'forecast' in forecast.columns:
                    print(forecast[['forecast', 'lower_ci', 'upper_ci']].round(2))
                else:
                    print(forecast.round(2))
                    
            except Exception as e:
                print(f"⚠ Forecast failed for {model_name}: {e}")
    
    # Step 7: Generate comprehensive professional report
    print("\n" + "="*60)
    print("STEP 7: PROFESSIONAL REPORT")
    print("="*60)
    
    report = agent.generate_report()
    print(report)
    
    # Additional insights
    print("\n" + "="*60)
    print("ADDITIONAL INSIGHTS")
    print("="*60)
    
    print(f"\n📈 Analysis Summary:")
    print(f"   • Datasets analyzed: {len(agent.datasets)}")
    print(f"   • Models built: {len(models_built)}")
    print(f"   • Forecasts generated: {len(forecasts_generated)}")
    print(f"   • Analysis steps completed: {len(agent.analysis_history)}")
    
    if validation['quality_score'] >= 80:
        print(f"   • Data quality: Excellent ({validation['quality_score']:.1f}/100)")
    elif validation['quality_score'] >= 60:
        print(f"   • Data quality: Good ({validation['quality_score']:.1f}/100)")
    else:
        print(f"   • Data quality: Needs improvement ({validation['quality_score']:.1f}/100)")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    
    if not fred_api_key:
        print("   • Get a FRED API key for access to real economic data")
    
    if validation['quality_score'] < 90:
        print("   • Review data quality issues and consider additional cleaning")
    
    non_stationary_vars = [var for var, result in stationarity_results.items() 
                          if 'non-stationary' in result.get('interpretation', '')]
    if non_stationary_vars:
        print(f"   • Consider differencing for non-stationary variables: {', '.join(non_stationary_vars)}")
    
    if len(models_built) > 2:
        print("   • Use cross-validation for more robust model selection")
        print("   • Consider ensemble methods for improved forecasting accuracy")
    
    print(f"\n🎯 Next Steps:")
    print("   • Implement rolling window validation for time series models")
    print("   • Add seasonal decomposition analysis")
    print("   • Develop early warning indicators for economic cycles")
    print("   • Create automated model retraining pipeline")
    
    print("\n" + "="*80)
    print("PROFESSIONAL ANALYSIS COMPLETED")
    print("="*80)


if __name__ == "__main__":
    run_professional_analysis()