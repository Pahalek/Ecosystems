"""
Enhanced Data Fetcher for Missing Econometric Variables
Extends existing APIs to fill data gaps in Amazon ecosystem analysis
"""
import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta
import fredapi

# Load environment variables
load_dotenv()

class EnhancedDataFetcher:
    def __init__(self):
        self.fred_api_key = os.getenv('FRED_API_KEY')
        if self.fred_api_key:
            self.fred = fredapi.Fred(api_key=self.fred_api_key)
        
    def fetch_competitor_cloud_data(self):
        """Fetch competitor data for cloud services comparison"""
        print("🔍 Fetching competitor cloud data...")
        
        # Microsoft (Azure competitor)
        msft = yf.Ticker("MSFT")
        msft_data = {
            'stock_data': msft.history(period="5y"),
            'info': msft.info,
            'financials': msft.financials,
            'quarterly_financials': msft.quarterly_financials
        }
        
        # Google (Google Cloud competitor)  
        googl = yf.Ticker("GOOGL")
        googl_data = {
            'stock_data': googl.history(period="5y"),
            'info': googl.info,
            'financials': googl.financials,
            'quarterly_financials': googl.quarterly_financials
        }
        
        return {
            'microsoft': msft_data,
            'google': googl_data
        }
    
    def fetch_ecommerce_industry_data(self):
        """Fetch e-commerce industry context data"""
        print("🛒 Fetching e-commerce industry data...")
        
        # E-commerce related stocks for industry context
        ecommerce_tickers = {
            'walmart': 'WMT',
            'target': 'TGT', 
            'shopify': 'SHOP',
            'alibaba': 'BABA',
            'ebay': 'EBAY'
        }
        
        ecommerce_data = {}
        for company, ticker in ecommerce_tickers.items():
            try:
                stock = yf.Ticker(ticker)
                ecommerce_data[company] = {
                    'stock_data': stock.history(period="3y"),
                    'info': stock.info
                }
            except Exception as e:
                print(f"⚠️ Could not fetch {company} data: {e}")
        
        return ecommerce_data
    
    def fetch_economic_indicators(self):
        """Fetch relevant economic indicators from FRED"""
        if not self.fred_api_key:
            print("⚠️ FRED API key not found. Economic indicators will be limited.")
            return {}
        
        print("📊 Fetching economic indicators...")
        
        # Key economic indicators for e-commerce/tech analysis
        fred_series = {
            'consumer_sentiment': 'UMCSENT',  # Consumer sentiment
            'retail_sales': 'RSXFS',  # Retail sales excluding food services
            'ecommerce_sales': 'ECOMSA',  # E-commerce retail sales
            'personal_consumption': 'PCE',  # Personal consumption expenditures
            'unemployment_rate': 'UNRATE',  # Unemployment rate
            'gdp_growth': 'GDP',  # Gross Domestic Product
            'inflation_rate': 'CPIAUCSL',  # Consumer Price Index
            'interest_rates': 'FEDFUNDS',  # Federal funds rate
            'tech_spending': 'A006RC1Q027SBEA'  # Technology equipment spending
        }
        
        economic_data = {}
        start_date = datetime.now() - timedelta(days=5*365)  # 5 years of data
        
        for indicator, series_id in fred_series.items():
            try:
                data = self.fred.get_series(series_id, start_date=start_date)
                economic_data[indicator] = data
                print(f"✅ Fetched {indicator}: {len(data)} observations")
            except Exception as e:
                print(f"⚠️ Could not fetch {indicator}: {e}")
        
        return economic_data
    
    def fetch_cloud_market_data(self):
        """Fetch cloud computing market context data"""
        print("☁️ Fetching cloud market data...")
        
        # Cloud-related stocks for market context
        cloud_tickers = {
            'salesforce': 'CRM',
            'oracle': 'ORCL',
            'ibm': 'IBM',
            'vmware': 'VMW',
            'snowflake': 'SNOW',
            'mongodb': 'MDB',
            'datadog': 'DDOG'
        }
        
        cloud_data = {}
        for company, ticker in cloud_tickers.items():
            try:
                stock = yf.Ticker(ticker)
                cloud_data[company] = {
                    'stock_data': stock.history(period="3y"),
                    'info': stock.info
                }
            except Exception as e:
                print(f"⚠️ Could not fetch {company} data: {e}")
        
        return cloud_data
    
    def fetch_streaming_competitors(self):
        """Fetch streaming service competitors for Prime Video analysis"""
        print("🎬 Fetching streaming market data...")
        
        streaming_tickers = {
            'netflix': 'NFLX',
            'disney': 'DIS',
            'warner_discovery': 'WBD',
            'paramount': 'PARA',
            'roku': 'ROKU'
        }
        
        streaming_data = {}
        for company, ticker in streaming_tickers.items():
            try:
                stock = yf.Ticker(ticker)
                streaming_data[company] = {
                    'stock_data': stock.history(period="3y"),
                    'info': stock.info
                }
            except Exception as e:
                print(f"⚠️ Could not fetch {company} data: {e}")
        
        return streaming_data
    
    def fetch_advertising_market_data(self):
        """Fetch digital advertising market context"""
        print("📺 Fetching advertising market data...")
        
        # Digital advertising related companies
        ad_tickers = {
            'meta': 'META',  # Facebook/Instagram ads
            'google': 'GOOGL',  # Google Ads  
            'trade_desk': 'TTD',  # Programmatic advertising
            'magnite': 'MGNI',  # Ad tech
            'pubmatic': 'PUBM'  # Publisher monetization
        }
        
        ad_data = {}
        for company, ticker in ad_tickers.items():
            try:
                stock = yf.Ticker(ticker)
                ad_data[company] = {
                    'stock_data': stock.history(period="3y"),
                    'info': stock.info
                }
            except Exception as e:
                print(f"⚠️ Could not fetch {company} data: {e}")
        
        return ad_data
    
    def fetch_comprehensive_dataset(self):
        """Fetch all enhanced data for comprehensive analysis"""
        print("🚀 Fetching comprehensive dataset for Amazon econometric analysis...")
        
        comprehensive_data = {
            'competitors_cloud': self.fetch_competitor_cloud_data(),
            'ecommerce_industry': self.fetch_ecommerce_industry_data(),
            'economic_indicators': self.fetch_economic_indicators(),
            'cloud_market': self.fetch_cloud_market_data(),
            'streaming_competitors': self.fetch_streaming_competitors(),
            'advertising_market': self.fetch_advertising_market_data(),
            'metadata': {
                'fetch_date': datetime.now().isoformat(),
                'data_sources': ['Yahoo Finance', 'FRED'],
                'purpose': 'Amazon ecosystem econometric analysis'
            }
        }
        
        return comprehensive_data
    
    def save_enhanced_data(self, data, filename="enhanced_amazon_data.pkl"):
        """Save the enhanced dataset for analysis"""
        import pickle
        
        filepath = f"data_cache/{filename}"
        os.makedirs("data_cache", exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"💾 Enhanced dataset saved to: {filepath}")
        return filepath

def main():
    """Fetch enhanced dataset for Amazon econometric analysis"""
    fetcher = EnhancedDataFetcher()
    
    # Fetch comprehensive data
    enhanced_data = fetcher.fetch_comprehensive_dataset()
    
    # Save the data
    filepath = fetcher.save_enhanced_data(enhanced_data)
    
    # Summary
    print("\n📊 ENHANCED DATA SUMMARY")
    print("=" * 50)
    print(f"✅ Competitor cloud data: Microsoft, Google")
    print(f"✅ E-commerce industry: 5 major players")
    print(f"✅ Economic indicators: {len(enhanced_data['economic_indicators'])} series")
    print(f"✅ Cloud market context: 7 companies")
    print(f"✅ Streaming competitors: 5 companies")
    print(f"✅ Advertising market: 5 companies")
    print(f"💾 Data saved to: {filepath}")
    print("\n🎯 Your Amazon econometric model now has comprehensive market context!")

if __name__ == "__main__":
    main()
