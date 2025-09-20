"""
Amazon Ecosystem Econometric Analysis
Advanced econometric modeling of Amazon's business ecosystem using real data.
Combines financial, market, and economic data for comprehensive analysis.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf
from dotenv import load_dotenv

# Import econometric libraries
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load environment variables
load_dotenv()

class AmazonEconometricAnalysis:
    def __init__(self):
        self.data = {}
        self.models = {}
        self.results = {}
        
    def fetch_amazon_data(self):
        """Fetch comprehensive Amazon and market data."""
        print("🔍 Fetching Amazon ecosystem data...")
        
        # Amazon stock data
        amzn = yf.Ticker("AMZN")
        self.data['amzn_stock'] = amzn.history(period="5y")
        self.data['amzn_info'] = amzn.info
        
        # Market benchmarks
        sp500 = yf.Ticker("^GSPC")
        self.data['sp500'] = sp500.history(period="5y")
        
        # Sector ETF (Consumer Discretionary)
        xly = yf.Ticker("XLY")
        self.data['xly'] = xly.history(period="5y")
        
        # E-commerce competitors
        competitors = ['SHOP', 'EBAY', 'WMT', 'TGT']
        self.data['competitors'] = {}
        for ticker in competitors:
            try:
                comp = yf.Ticker(ticker)
                self.data['competitors'][ticker] = comp.history(period="5y")
            except Exception as e:
                print(f"Could not fetch {ticker}: {e}")
        
        # Economic indicators (using FRED data through yfinance if available)
        try:
            # Consumer confidence
            cci = yf.Ticker("CCI")  # Consumer Confidence Index
            self.data['cci'] = cci.history(period="5y")
        except:
            print("Consumer confidence data not available via yfinance")
        
        print(f"✅ Data fetched for {len(self.data)} datasets")
        
    def prepare_analysis_dataset(self):
        """Prepare consolidated dataset for econometric analysis."""
        print("📊 Preparing analysis dataset...")
        
        # Start with Amazon stock data
        df = self.data['amzn_stock'].copy()
        df = df.rename(columns={'Close': 'AMZN_Close', 'Volume': 'AMZN_Volume'})
        
        # Add market benchmark
        if 'sp500' in self.data:
            sp500_close = self.data['sp500']['Close'].reindex(df.index)
            df['SP500_Close'] = sp500_close
            df['AMZN_Beta'] = df['AMZN_Close'].pct_change().rolling(252).cov(sp500_close.pct_change()) / sp500_close.pct_change().rolling(252).var()
        
        # Add sector benchmark
        if 'xly' in self.data:
            df['XLY_Close'] = self.data['xly']['Close'].reindex(df.index)
        
        # Add competitor data
        for ticker, data in self.data.get('competitors', {}).items():
            if not data.empty:
                df[f'{ticker}_Close'] = data['Close'].reindex(df.index)
        
        # Calculate financial metrics
        df['AMZN_Returns'] = df['AMZN_Close'].pct_change()
        df['AMZN_Log_Returns'] = np.log(df['AMZN_Close'] / df['AMZN_Close'].shift(1))
        df['AMZN_Volatility'] = df['AMZN_Returns'].rolling(30).std() * np.sqrt(252)
        df['AMZN_MA_20'] = df['AMZN_Close'].rolling(20).mean()
        df['AMZN_MA_50'] = df['AMZN_Close'].rolling(50).mean()
        df['AMZN_RSI'] = self.calculate_rsi(df['AMZN_Close'])
        
        # Market metrics
        if 'SP500_Close' in df.columns:
            df['Market_Returns'] = df['SP500_Close'].pct_change()
            df['Excess_Returns'] = df['AMZN_Returns'] - df['Market_Returns']
        
        # Remove rows with NaN values
        df = df.dropna()
        
        self.data['analysis_df'] = df
        print(f"✅ Analysis dataset prepared: {len(df)} observations, {len(df.columns)} variables")
        
        return df
    
    def calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def time_series_analysis(self):
        """Perform time series analysis on Amazon stock."""
        print("📈 Performing time series analysis...")
        
        df = self.data['analysis_df']
        amzn_prices = df['AMZN_Close']
        
        # Decomposition
        decomposition = seasonal_decompose(amzn_prices, model='multiplicative', period=252)
        
        # ARIMA modeling
        try:
            # Fit ARIMA model
            model = ARIMA(amzn_prices, order=(1, 1, 1))
            fitted_model = model.fit()
            
            # Forecast
            forecast = fitted_model.forecast(steps=30)
            
            self.models['arima'] = fitted_model
            self.results['forecast'] = forecast
            
            print(f"✅ ARIMA model fitted. AIC: {fitted_model.aic:.2f}")
            
        except Exception as e:
            print(f"❌ ARIMA modeling failed: {e}")
        
        # Store decomposition results
        self.results['decomposition'] = {
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid
        }
        
    def regression_analysis(self):
        """Perform regression analysis."""
        print("📊 Performing regression analysis...")
        
        df = self.data['analysis_df']
        
        # Prepare variables for regression
        y = df['AMZN_Returns'].dropna()
        
        # Independent variables
        X_vars = []
        if 'Market_Returns' in df.columns:
            X_vars.append('Market_Returns')
        if 'XLY_Close' in df.columns:
            X_vars.append('XLY_Close')
        
        # Add competitor returns
        for ticker in self.data.get('competitors', {}).keys():
            col_name = f'{ticker}_Close'
            if col_name in df.columns:
                df[f'{ticker}_Returns'] = df[col_name].pct_change()
                X_vars.append(f'{ticker}_Returns')
        
        if X_vars:
            X = df[X_vars].dropna()
            
            # Align X and y
            common_index = X.index.intersection(y.index)
            X = X.loc[common_index]
            y = y.loc[common_index]
            
            if len(X) > 0:
                # Add constant
                X = sm.add_constant(X)
                
                # Fit regression model
                model = sm.OLS(y, X).fit()
                self.models['regression'] = model
                
                print(f"✅ Regression model fitted. R²: {model.rsquared:.4f}")
                print(f"📊 Significant variables (p < 0.05):")
                
                for var, p_val in model.pvalues.items():
                    if p_val < 0.05:
                        coef = model.params[var]
                        print(f"   • {var}: coefficient = {coef:.4f}, p-value = {p_val:.4f}")
            else:
                print("❌ Insufficient data for regression analysis")
        else:
            print("❌ No suitable independent variables found for regression")
    
    def machine_learning_analysis(self):
        """Perform machine learning analysis."""
        print("🤖 Performing machine learning analysis...")
        
        df = self.data['analysis_df']
        
        # Prepare features for ML
        feature_cols = []
        
        # Technical indicators
        if 'AMZN_Volatility' in df.columns:
            feature_cols.append('AMZN_Volatility')
        if 'AMZN_RSI' in df.columns:
            feature_cols.append('AMZN_RSI')
        
        # Market features
        if 'Market_Returns' in df.columns:
            feature_cols.append('Market_Returns')
        if 'AMZN_Beta' in df.columns:
            feature_cols.append('AMZN_Beta')
        
        # Add lagged returns
        for lag in [1, 2, 3, 5]:
            col_name = f'AMZN_Returns_Lag_{lag}'
            df[col_name] = df['AMZN_Returns'].shift(lag)
            feature_cols.append(col_name)
        
        # Target: next day return
        df['AMZN_Returns_Next'] = df['AMZN_Returns'].shift(-1)
        
        # Prepare data
        X = df[feature_cols].dropna()
        y = df['AMZN_Returns_Next'].loc[X.index].dropna()
        
        # Ensure X and y have the same index
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        
        # Remove any remaining NaN values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[mask]
        y = y[mask]
        
        if len(X) > 100:  # Need sufficient data
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Train Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            
            # Predictions
            y_pred = rf.predict(X_test)
            
            # Metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.models['random_forest'] = rf
            self.results['ml_metrics'] = {
                'mse': mse,
                'r2': r2,
                'feature_importance': dict(zip(feature_cols, rf.feature_importances_))
            }
            
            print(f"✅ Random Forest model trained. R²: {r2:.4f}, MSE: {mse:.6f}")
            print("📊 Top 3 important features:")
            
            importance_sorted = sorted(
                self.results['ml_metrics']['feature_importance'].items(),
                key=lambda x: x[1], reverse=True
            )
            
            for feature, importance in importance_sorted[:3]:
                print(f"   • {feature}: {importance:.4f}")
        else:
            print("❌ Insufficient data for machine learning analysis")
    
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report."""
        print("\n" + "="*60)
        print("🎯 AMAZON ECOSYSTEM ECONOMETRIC ANALYSIS REPORT")
        print("="*60)
        
        # Company overview
        info = self.data.get('amzn_info', {})
        print(f"\n📈 COMPANY OVERVIEW")
        print(f"Company: {info.get('shortName', 'Amazon.com, Inc.')}")
        print(f"Market Cap: ${info.get('marketCap', 0):,.0f}")
        print(f"Revenue (TTM): ${info.get('totalRevenue', 0):,.0f}")
        print(f"Sector: {info.get('sector', 'Consumer Cyclical')}")
        
        # Stock performance
        df = self.data.get('analysis_df')
        if df is not None and not df.empty:
            current_price = df['AMZN_Close'].iloc[-1]
            year_ago_price = df['AMZN_Close'].iloc[-252] if len(df) >= 252 else df['AMZN_Close'].iloc[0]
            annual_return = (current_price / year_ago_price - 1) * 100
            annual_volatility = df['AMZN_Returns'].std() * np.sqrt(252) * 100
            
            print(f"\n📊 PERFORMANCE METRICS")
            print(f"Current Price: ${current_price:.2f}")
            print(f"Annual Return: {annual_return:.2f}%")
            print(f"Annual Volatility: {annual_volatility:.2f}%")
            print(f"Sharpe Ratio: {annual_return / annual_volatility:.2f}")
        
        # Model results
        if 'regression' in self.models:
            model = self.models['regression']
            print(f"\n🔬 REGRESSION ANALYSIS")
            print(f"R-squared: {model.rsquared:.4f}")
            print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
            print(f"F-statistic: {model.fvalue:.2f} (p-value: {model.f_pvalue:.4f})")
        
        if 'ml_metrics' in self.results:
            metrics = self.results['ml_metrics']
            print(f"\n🤖 MACHINE LEARNING ANALYSIS")
            print(f"Random Forest R²: {metrics['r2']:.4f}")
            print(f"Mean Squared Error: {metrics['mse']:.6f}")
        
        # Investment insights
        print(f"\n💡 INVESTMENT INSIGHTS")
        if df is not None and not df.empty:
            recent_rsi = df['AMZN_RSI'].iloc[-1] if 'AMZN_RSI' in df.columns else None
            if recent_rsi is not None:
                if recent_rsi > 70:
                    print("• RSI indicates potentially overbought conditions")
                elif recent_rsi < 30:
                    print("• RSI indicates potentially oversold conditions")
                else:
                    print("• RSI indicates neutral momentum")
            
            if 'AMZN_Beta' in df.columns:
                recent_beta = df['AMZN_Beta'].iloc[-1]
                if not np.isnan(recent_beta):
                    print(f"• Beta: {recent_beta:.2f} (market sensitivity)")
        
        print(f"\n⚠️  DISCLAIMER: This analysis is for educational purposes only.")
        print(f"   Not investment advice. Past performance doesn't guarantee future results.")

def main():
    """Main analysis workflow."""
    print("🚀 Amazon Ecosystem Econometric Analysis")
    print("="*50)
    
    # Initialize analysis
    analysis = AmazonEconometricAnalysis()
    
    # Fetch data
    analysis.fetch_amazon_data()
    
    # Prepare dataset
    analysis.prepare_analysis_dataset()
    
    # Perform analyses
    analysis.time_series_analysis()
    analysis.regression_analysis()
    analysis.machine_learning_analysis()
    
    # Generate report
    analysis.generate_comprehensive_report()
    
    # Create visualizations
    create_visualizations(analysis)

def create_visualizations(analysis):
    """Create comprehensive visualizations."""
    print(f"\n📊 Creating visualizations...")
    
    df = analysis.data.get('analysis_df')
    if df is None or df.empty:
        print("❌ No data available for visualization")
        return
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Amazon Ecosystem Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Stock price and moving averages
    ax1 = axes[0, 0]
    ax1.plot(df.index, df['AMZN_Close'], label='AMZN Close', linewidth=2)
    if 'AMZN_MA_20' in df.columns:
        ax1.plot(df.index, df['AMZN_MA_20'], label='20-day MA', alpha=0.7)
    if 'AMZN_MA_50' in df.columns:
        ax1.plot(df.index, df['AMZN_MA_50'], label='50-day MA', alpha=0.7)
    ax1.set_title('Amazon Stock Price & Moving Averages')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Returns distribution
    ax2 = axes[0, 1]
    if 'AMZN_Returns' in df.columns:
        ax2.hist(df['AMZN_Returns'].dropna(), bins=50, alpha=0.7, edgecolor='black')
        ax2.axvline(df['AMZN_Returns'].mean(), color='red', linestyle='--', label='Mean')
        ax2.set_title('Amazon Daily Returns Distribution')
        ax2.set_xlabel('Daily Returns')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Volatility over time
    ax3 = axes[1, 0]
    if 'AMZN_Volatility' in df.columns:
        ax3.plot(df.index, df['AMZN_Volatility'], color='orange', linewidth=2)
        ax3.set_title('Amazon Volatility (30-day rolling)')
        ax3.set_ylabel('Volatility')
        ax3.grid(True, alpha=0.3)
    
    # 4. Performance comparison
    ax4 = axes[1, 1]
    # Normalize prices to show relative performance
    if 'SP500_Close' in df.columns and 'XLY_Close' in df.columns:
        amzn_norm = df['AMZN_Close'] / df['AMZN_Close'].iloc[0] * 100
        sp500_norm = df['SP500_Close'] / df['SP500_Close'].iloc[0] * 100
        xly_norm = df['XLY_Close'] / df['XLY_Close'].iloc[0] * 100
        
        ax4.plot(df.index, amzn_norm, label='AMZN', linewidth=2)
        ax4.plot(df.index, sp500_norm, label='S&P 500', alpha=0.7)
        ax4.plot(df.index, xly_norm, label='Consumer Disc. (XLY)', alpha=0.7)
        ax4.set_title('Relative Performance (Normalized to 100)')
        ax4.set_ylabel('Normalized Price')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = 'd:/Econometric/Ecosystems/amazon_econometric_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Analysis dashboard saved to: amazon_econometric_analysis.png")
    plt.show()

if __name__ == "__main__":
    main()
