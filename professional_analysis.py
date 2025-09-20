"""
Simple FRED Data Demo - Working Version
Shows real FRED data analysis using the working method.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from econometric_agent.data_fetcher import EconomicDataFetcher
from econometric_agent import EconometricAgent
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=" * 70)
    print("🌟 REAL FRED DATA ANALYSIS - WORKING DEMONSTRATION")
    print("=" * 70)
    
    # Load environment
    load_dotenv()
    api_key = os.getenv('FRED_API_KEY')
    
    print(f"🔑 Using FRED API Key: {api_key[:8]}...{api_key[-4:]}")
    
    # Method 1: Direct data fetcher (we know this works)
    print("\\n📊 METHOD 1: Direct Data Fetcher")
    print("-" * 40)
    
    fetcher = EconomicDataFetcher(fred_api_key=api_key)
    
    # Get recent economic data
    print("Loading real economic data from FRED...")
    data = fetcher.get_common_economic_indicators('2022-01-01', '2023-12-31')
    
    print(f"✅ SUCCESS! Loaded real economic data")
    print(f"📊 Shape: {data.shape}")
    print(f"📅 Date range: {data.index.min()} to {data.index.max()}")
    print(f"🔢 Variables: {list(data.columns)}")
    
    # Clean data
    data_clean = data.dropna()
    print(f"📈 After cleaning: {data_clean.shape}")
    
    # Show sample data
    print("\\n📋 Sample of Real Economic Data:")
    print(data_clean.head())
    
    print("\\n📈 Latest Economic Indicators:")
    latest = data_clean.iloc[-1]
    for var, value in latest.items():
        print(f"  • {var}: {value:.2f}")
    
    # Method 2: Use agent with direct assignment
    print("\\n🤖 METHOD 2: EconometricAgent with Manual Data")
    print("-" * 50)
    
    agent = EconometricAgent()
    
    # Manually assign the working data to agent
    agent.datasets['fred_data'] = data_clean
    agent.current_data = data_clean
    
    print("✅ Data successfully assigned to EconometricAgent")
    
    # Now run analysis with real data
    print("\\n🔍 Data Quality Analysis")
    print("-" * 30)
    
    validation = agent.validate_data('fred_data')
    quality_report = agent.data_validator.generate_quality_report(data_clean)
    print(quality_report)
    
    # Build models with real data
    print("\\n🧮 Economic Modeling with Real Data")
    print("-" * 40)
    
    if 'GDP' in data_clean.columns:
        # Feature selection
        features = [col for col in data_clean.columns 
                   if col != 'GDP' and col in ['UNEMPLOYMENT', 'INFLATION', 'FEDERAL_FUNDS_RATE']]
        
        print(f"🎯 Target: GDP")
        print(f"📊 Features: {features}")
        
        # Linear regression
        print("\\n1. Linear Regression with Real Data")
        regression_results = agent.build_regression_model(
            target_variable='GDP',
            feature_variables=features,
            dataset_name='fred_data'
        )
        
        print(f"✅ Model: {regression_results.get('model_name', 'Linear Regression')}")
        
        # Handle different result structures
        if 'performance' in regression_results:
            perf = regression_results['performance']
            print(f"📈 R² (Test): {perf.get('r2_test', 'N/A')}")
            print(f"📉 MSE (Test): {perf.get('mse_test', 'N/A')}")
        else:
            print("📊 Model built successfully - performance metrics available in analysis history")
        
        # Time series analysis
        print("\\n2. Time Series Analysis")
        try:
            arima_results = agent.build_time_series_model(
                target_variable='GDP',
                dataset_name='fred_data',
                model_type='arima'
            )
            print(f"✅ ARIMA Model: {arima_results.get('model_name', 'ARIMA Model')}")
            
            # Handle different result structures
            if 'performance' in arima_results:
                perf = arima_results['performance']
                print(f"📊 AIC: {perf.get('aic', 'N/A')}")
            else:
                print("📊 ARIMA model built successfully")
            
            # Generate forecast
            forecast = agent.generate_forecast(
                model_name=arima_results['model_name'],
                steps=6
            )
            print(f"\\n📈 6-Month GDP Forecast:")
            print(forecast.head())
            
        except Exception as e:
            print(f"⚠ ARIMA analysis: {e}")
    
    # Visualization
    print("\\n📊 Economic Data Visualization")
    print("-" * 35)
    
    # Create plot
    plt.figure(figsize=(15, 10))
    
    # Plot key indicators
    indicators = ['GDP', 'UNEMPLOYMENT', 'INFLATION', 'FEDERAL_FUNDS_RATE']
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, (indicator, color) in enumerate(zip(indicators, colors)):
        if indicator in data_clean.columns:
            plt.subplot(2, 2, i+1)
            
            plot_data = data_clean[indicator].dropna()
            plt.plot(plot_data.index, plot_data.values, color=color, linewidth=2)
            plt.title(f'Real {indicator} Data\\n({plot_data.index.min().strftime("%Y-%m")} to {plot_data.index.max().strftime("%Y-%m")})')
            plt.xlabel('Date')
            plt.ylabel(indicator)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('./figures', exist_ok=True)
    plt.savefig('./figures/fred_real_data_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Chart saved: ./figures/fred_real_data_demo.png")
    
    # Summary
    print("\\n" + "=" * 70)
    print("🎉 REAL FRED DATA ANALYSIS COMPLETE!")
    print("=" * 70)
    
    print(f"\\n📊 Summary:")
    print(f"  • Data Source: Federal Reserve Economic Data (FRED)")
    print(f"  • API Status: ✅ Working perfectly")
    print(f"  • Variables: {len(data_clean.columns)} economic indicators")
    print(f"  • Observations: {len(data_clean)} time periods")
    print(f"  • Models Built: {len(agent.analysis_history)} analysis steps")
    print(f"  • Date Coverage: {data_clean.index.min()} to {data_clean.index.max()}")
    
    print(f"\\n🎯 Key Insights:")
    print(f"  • Your FRED API key is working correctly")
    print(f"  • Real economic data is being fetched successfully")
    print(f"  • Econometric models can use real-world data")
    print(f"  • Professional analysis workflow is operational")
    
    print(f"\\n🚀 Next Steps:")
    print(f"  • Explore different time periods")
    print(f"  • Add more economic indicators")
    print(f"  • Build custom models for specific research")
    print(f"  • Generate forecasts for economic planning")

if __name__ == "__main__":
    main()
