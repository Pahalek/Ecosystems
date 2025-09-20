"""
Amazon Econometric Model with Investor Report Integration
Combines API data with parsed investor report data for comprehensive analysis.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import yfinance as yf
from dotenv import load_dotenv

# Import our existing modules
from amazon_business_intelligence import AmazonDataFetcher
from amazon_investor_parser import AmazonInvestorReportsParser

# Econometric libraries
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load environment variables
load_dotenv()

class EnhancedAmazonEconometricModel:
    def __init__(self):
        self.api_data = {}
        self.investor_data = {}
        self.integrated_dataset = None
        self.models = {}
        self.results = {}
        
        # Initialize data fetchers
        self.api_fetcher = AmazonDataFetcher()
        self.report_parser = AmazonInvestorReportsParser()
        
    def fetch_api_data(self):
        """Fetch data from APIs (Yahoo Finance, Alpha Vantage, etc.)."""
        print("📊 Fetching API data...")
        
        # Get comprehensive API data
        self.api_fetcher.fetch_yahoo_finance_data()
        self.api_fetcher.fetch_alpha_vantage_data()
        
        self.api_data = self.api_fetcher.data
        print("✅ API data fetched successfully")
        
    def load_investor_report_data(self):
        """Load and process investor report data."""
        print("📄 Loading investor report data...")
        
        # Check if parsed data exists
        parsed_data_dir = Path("investor_reports/parsed_data")
        
        if not parsed_data_dir.exists():
            print("⚠️  No parsed investor data found. Processing reports...")
            self.report_parser.process_all_files()
            self.report_parser.generate_metrics_summary()
            self.report_parser.export_to_dataframe()
        
        # Load parsed metrics
        metrics_file = parsed_data_dir / "metrics_summary.json"
        csv_file = parsed_data_dir / "extracted_metrics.csv"
        
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                self.investor_data['summary'] = json.load(f)
            print("✅ Loaded investor metrics summary")
        
        if csv_file.exists():
            self.investor_data['metrics_df'] = pd.read_csv(csv_file)
            print("✅ Loaded investor metrics DataFrame")
        
        if not self.investor_data:
            print("❌ No investor report data available")
            return False
        
        return True
        
    def create_integrated_dataset(self):
        """Create integrated dataset combining API and investor report data."""
        print("🔗 Creating integrated dataset...")
        
        if not self.api_data:
            print("❌ No API data available")
            return None
            
        # Start with stock price data
        if 'yahoo_finance' in self.api_data:
            stock_data = self.api_data['yahoo_finance']['history'].copy()
            
            # Add basic financial metrics
            info = self.api_data['yahoo_finance']['info']
            
            # Add quarterly/annual data points where available
            enhanced_data = stock_data.copy()
            enhanced_data['Quarter'] = enhanced_data.index.to_period('Q')
            enhanced_data['Year'] = enhanced_data.index.year
            
            # Extract segment data from investor reports
            segment_data = self.extract_segment_metrics()
            
            # Add investor report metrics
            if segment_data:
                enhanced_data = self.merge_segment_data(enhanced_data, segment_data)
            
            self.integrated_dataset = enhanced_data
            print(f"✅ Integrated dataset created: {len(enhanced_data)} observations")
            return enhanced_data
        
        return None
    
    def extract_segment_metrics(self):
        """Extract AWS, Geographic, and Advertising metrics from investor reports."""
        if 'metrics_df' not in self.investor_data:
            return None
            
        df = self.investor_data['metrics_df']
        segment_metrics = {}
        
        # Process AWS metrics
        aws_data = df[df['metric_type'] == 'aws_data']
        if not aws_data.empty:
            segment_metrics['aws'] = self.parse_financial_values(aws_data)
        
        # Process Geographic metrics
        geo_data = df[df['metric_type'] == 'geographic_data']
        if not geo_data.empty:
            segment_metrics['geographic'] = self.parse_financial_values(geo_data)
        
        # Process Advertising metrics
        ad_data = df[df['metric_type'] == 'advertising_data']
        if not ad_data.empty:
            segment_metrics['advertising'] = self.parse_financial_values(ad_data)
        
        # Process Prime metrics
        prime_data = df[df['metric_type'] == 'prime_data']
        if not prime_data.empty:
            segment_metrics['prime'] = self.parse_financial_values(prime_data)
        
        return segment_metrics
    
    def parse_financial_values(self, data):
        """Parse financial values from extracted text."""
        parsed_values = []
        
        for _, row in data.iterrows():
            value_str = row['extracted_value']
            if pd.notna(value_str):
                try:
                    # Clean and convert financial values
                    cleaned = str(value_str).replace(',', '').replace('$', '').replace('%', '')
                    
                    # Handle billions/millions
                    if 'billion' in row['matched_text'].lower():
                        value = float(cleaned) * 1e9
                    elif 'million' in row['matched_text'].lower():
                        value = float(cleaned) * 1e6
                    else:
                        value = float(cleaned)
                    
                    parsed_values.append({
                        'value': value,
                        'source': row['source_file'],
                        'context': row['matched_text'],
                        'raw_value': value_str
                    })
                    
                except (ValueError, TypeError):
                    continue
        
        return parsed_values
    
    def merge_segment_data(self, stock_data, segment_data):
        """Merge segment data with stock data."""
        enhanced_data = stock_data.copy()
        
        # Add segment indicators based on available data
        if 'aws' in segment_data and segment_data['aws']:
            # Add AWS revenue trend (simplified)
            aws_values = [item['value'] for item in segment_data['aws']]
            if aws_values:
                enhanced_data['AWS_Revenue_Indicator'] = np.mean(aws_values)
        
        if 'geographic' in segment_data and segment_data['geographic']:
            geo_values = [item['value'] for item in segment_data['geographic']]
            if geo_values:
                enhanced_data['Geographic_Revenue_Indicator'] = np.mean(geo_values)
        
        if 'advertising' in segment_data and segment_data['advertising']:
            ad_values = [item['value'] for item in segment_data['advertising']]
            if ad_values:
                enhanced_data['Advertising_Revenue_Indicator'] = np.mean(ad_values)
        
        return enhanced_data
    
    def perform_enhanced_analysis(self):
        """Perform econometric analysis with integrated data."""
        if self.integrated_dataset is None:
            print("❌ No integrated dataset available")
            return
            
        print("🔬 Performing enhanced econometric analysis...")
        
        df = self.integrated_dataset.copy()
        
        # Prepare variables
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility'] = df['Returns'].rolling(30).std() * np.sqrt(252)
        
        # Enhanced regression with segment data
        y = df['Returns'].dropna()
        
        # Base variables
        X_vars = ['Volume']
        
        # Add segment variables if available
        segment_vars = [col for col in df.columns if 'Indicator' in col]
        X_vars.extend(segment_vars)
        
        # Create feature matrix
        X = df[X_vars].dropna()
        
        # Align indices
        common_index = X.index.intersection(y.index)
        if len(common_index) > 50:  # Need sufficient data
            X = X.loc[common_index]
            y = y.loc[common_index]
            
            # Add constant
            X = sm.add_constant(X)
            
            # Fit enhanced model
            model = sm.OLS(y, X).fit()
            self.models['enhanced_regression'] = model
            
            print("✅ Enhanced regression model fitted")
            print(f"R-squared: {model.rsquared:.4f}")
            
            # Show segment impact
            for var in segment_vars:
                if var in model.params.index:
                    coef = model.params[var]
                    p_val = model.pvalues[var]
                    print(f"📊 {var}: coefficient = {coef:.6f}, p-value = {p_val:.4f}")
        
        self.results['enhanced_analysis'] = {
            'model_summary': model.summary() if 'model' in locals() else None,
            'segment_vars_used': segment_vars,
            'total_observations': len(df)
        }
    
    def generate_comprehensive_report(self):
        """Generate comprehensive report with investor data integration."""
        print("\n" + "="*80)
        print("🎯 ENHANCED AMAZON ECONOMETRIC ANALYSIS WITH INVESTOR DATA")
        print("="*80)
        
        # API Data Summary
        if self.api_data:
            if 'yahoo_finance' in self.api_data:
                info = self.api_data['yahoo_finance']['info']
                print(f"\n📊 FINANCIAL OVERVIEW (API Data)")
                print(f"Market Cap: ${info.get('marketCap', 0):,.0f}")
                print(f"Revenue (TTM): ${info.get('totalRevenue', 0):,.0f}")
                print(f"Current Price: ${info.get('currentPrice', 0):.2f}")
        
        # Investor Report Data Summary
        if self.investor_data:
            print(f"\n📄 INVESTOR REPORT DATA")
            
            if 'summary' in self.investor_data:
                summary = self.investor_data['summary']
                total_files = summary.get('total_files_processed', 0)
                print(f"Files Processed: {total_files}")
                
                # Show metrics found
                metric_types = ['aws_metrics', 'geographic_metrics', 'advertising_metrics', 'prime_metrics']
                for metric_type in metric_types:
                    count = len(summary.get(metric_type, []))
                    if count > 0:
                        print(f"✅ {metric_type.replace('_', ' ').title()}: {count} data points")
        
        # Integration Results
        if self.integrated_dataset is not None:
            print(f"\n🔗 INTEGRATED DATASET")
            print(f"Total Observations: {len(self.integrated_dataset)}")
            
            segment_cols = [col for col in self.integrated_dataset.columns if 'Indicator' in col]
            if segment_cols:
                print(f"Segment Variables Added: {len(segment_cols)}")
                for col in segment_cols:
                    print(f"  • {col}")
            else:
                print("⚠️  No segment variables integrated (may need more investor data)")
        
        # Model Results
        if 'enhanced_regression' in self.models:
            model = self.models['enhanced_regression']
            print(f"\n🔬 ENHANCED MODEL RESULTS")
            print(f"R-squared: {model.rsquared:.4f}")
            print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
            print(f"Observations: {model.nobs}")
        
        print(f"\n💡 DATA INTEGRATION STATUS:")
        api_status = "✅ Connected" if self.api_data else "❌ Not available"
        investor_status = "✅ Integrated" if self.investor_data else "❌ No files processed"
        
        print(f"API Data Sources: {api_status}")
        print(f"Investor Report Data: {investor_status}")
        
        if not self.investor_data:
            print(f"\n📥 TO ADD INVESTOR REPORT DATA:")
            print(f"1. Download files from https://ir.aboutamazon.com/")
            print(f"2. Place in investor_reports/ directory")
            print(f"3. Run this analysis again")

def main():
    """Main analysis workflow with investor data integration."""
    print("🚀 ENHANCED AMAZON ECONOMETRIC ANALYSIS")
    print("With Investor Report Integration")
    print("="*60)
    
    # Initialize enhanced model
    model = EnhancedAmazonEconometricModel()
    
    # Fetch API data
    model.fetch_api_data()
    
    # Load investor report data
    model.load_investor_report_data()
    
    # Create integrated dataset
    model.create_integrated_dataset()
    
    # Perform enhanced analysis
    model.perform_enhanced_analysis()
    
    # Generate comprehensive report
    model.generate_comprehensive_report()
    
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"Your environment now combines API data with investor report insights.")

if __name__ == "__main__":
    main()
