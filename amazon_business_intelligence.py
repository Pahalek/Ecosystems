"""
Amazon Business Intelligence Dashboard
Fetches and analyzes real Amazon (AMZN) business data from multiple sources.
Requires API keys for Alpha Vantage and Quandl (add to .env file).
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import Alpha Vantage and Quandl
try:
    from alpha_vantage.fundamentaldata import FundamentalData
    ALPHA_VANTAGE_AVAILABLE = True
except ImportError:
    ALPHA_VANTAGE_AVAILABLE = False

try:
    import quandl
    QUANDL_AVAILABLE = True
except ImportError:
    QUANDL_AVAILABLE = False

# Load environment variables
load_dotenv()

class AmazonDataFetcher:
    def __init__(self):
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.quandl_key = os.getenv("QUANDL_API_KEY")
        self.data = {}
        
    def fetch_yahoo_finance_data(self):
        """Fetch comprehensive Amazon data from Yahoo Finance (free, no API key required)."""
        print("📊 Fetching Yahoo Finance data...")
        try:
            amzn = yf.Ticker("AMZN")
            
            # Basic info
            info = amzn.info
            
            # Historical stock prices (5 years)
            hist = amzn.history(period="5y")
            
            # Financial statements
            financials = amzn.financials
            balance_sheet = amzn.balance_sheet
            cashflow = amzn.cashflow
            
            # Key metrics
            earnings = amzn.earnings
            quarterly_earnings = amzn.quarterly_earnings
            
            self.data['yahoo_finance'] = {
                'info': info,
                'history': hist,
                'financials': financials,
                'balance_sheet': balance_sheet,
                'cashflow': cashflow,
                'earnings': earnings,
                'quarterly_earnings': quarterly_earnings
            }
            
            print(f"✅ Yahoo Finance: Retrieved {len(hist)} days of stock data")
            return True
            
        except Exception as e:
            print(f"❌ Yahoo Finance error: {e}")
            return False
    
    def fetch_alpha_vantage_data(self):
        """Fetch Alpha Vantage fundamental data."""
        if not ALPHA_VANTAGE_AVAILABLE:
            print("⚠️  Alpha Vantage library not available")
            return False
            
        if not self.alpha_vantage_key:
            print("⚠️  Alpha Vantage API key not found in .env file")
            return False
            
        print("📈 Fetching Alpha Vantage data...")
        try:
            fd = FundamentalData(self.alpha_vantage_key)
            
            # Get fundamental data
            balance_sheet, _ = fd.get_balance_sheet_annual("AMZN")
            income_statement, _ = fd.get_income_statement_annual("AMZN")
            cash_flow, _ = fd.get_cash_flow_annual("AMZN")
            
            self.data['alpha_vantage'] = {
                'balance_sheet': balance_sheet,
                'income_statement': income_statement,
                'cash_flow': cash_flow
            }
            
            print("✅ Alpha Vantage: Retrieved fundamental data")
            return True
            
        except Exception as e:
            print(f"❌ Alpha Vantage error: {e}")
            return False
    
    def fetch_quandl_data(self):
        """Fetch Quandl economic and market data."""
        if not QUANDL_AVAILABLE:
            print("⚠️  Quandl library not available")
            return False
            
        if not self.quandl_key:
            print("⚠️  Quandl API key not found in .env file")
            return False
            
        print("📊 Fetching Quandl data...")
        try:
            quandl.ApiConfig.api_key = self.quandl_key
            
            # Try to get relevant economic data
            # Note: Quandl datasets change over time, these are examples
            datasets = {}
            
            # Try different dataset codes
            try:
                # E-commerce retail sales
                datasets['ecommerce_sales'] = quandl.get("FRED/ECOMPCTNSA")
            except Exception as e:
                print(f"⚠️  E-commerce sales data not available: {e}")
            
            try:
                # Consumer spending
                datasets['consumer_spending'] = quandl.get("FRED/PCE")
            except Exception as e:
                print(f"⚠️  Consumer spending data not available: {e}")
                
            self.data['quandl'] = datasets
            print(f"✅ Quandl: Retrieved {len(datasets)} datasets")
            return True
            
        except Exception as e:
            print(f"❌ Quandl error: {e}")
            return False
    
    def generate_amazon_business_insights(self):
        """Generate insights from collected Amazon data."""
        if 'yahoo_finance' not in self.data:
            print("❌ No data available for analysis")
            return
            
        print("\n🔍 AMAZON BUSINESS INTELLIGENCE DASHBOARD")
        print("=" * 50)
        
        info = self.data['yahoo_finance']['info']
        history = self.data['yahoo_finance']['history']
        
        # Key business metrics
        print(f"Company: {info.get('shortName', 'Amazon.com, Inc.')}")
        print(f"Market Cap: ${info.get('marketCap', 0):,.0f}")
        print(f"Revenue (TTM): ${info.get('totalRevenue', 0):,.0f}")
        print(f"Employees: {info.get('fullTimeEmployees', 0):,}")
        print(f"Current Price: ${info.get('currentPrice', 0):.2f}")
        print(f"52-Week Range: ${info.get('fiftyTwoWeekLow', 0):.2f} - ${info.get('fiftyTwoWeekHigh', 0):.2f}")
        
        # Financial ratios
        print(f"\n📊 Financial Ratios:")
        print(f"P/E Ratio: {info.get('trailingPE', 0):.2f}")
        print(f"Price-to-Sales: {info.get('priceToSalesTrailing12Months', 0):.2f}")
        print(f"Profit Margin: {info.get('profitMargins', 0):.2%}")
        print(f"Return on Equity: {info.get('returnOnEquity', 0):.2%}")
        
        # Stock performance
        if not history.empty:
            recent_return = (history['Close'].iloc[-1] / history['Close'].iloc[-252] - 1) * 100
            volatility = history['Close'].pct_change().std() * np.sqrt(252) * 100
            
            print(f"\n📈 Stock Performance (1 Year):")
            print(f"Return: {recent_return:.2f}%")
            print(f"Volatility: {volatility:.2f}%")
        
        # Business segments (if available)
        business_summary = info.get('longBusinessSummary', '')
        if 'AWS' in business_summary:
            print(f"\n🏢 Key Business Segments:")
            print("• North America retail")
            print("• International retail")
            print("• Amazon Web Services (AWS)")
            print("• Advertising")
        
        return {
            'market_cap': info.get('marketCap', 0),
            'revenue': info.get('totalRevenue', 0),
            'employees': info.get('fullTimeEmployees', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'profit_margin': info.get('profitMargins', 0),
            'stock_data': history
        }

def main():
    """Main function to fetch and analyze Amazon data."""
    print("🚀 Amazon Ecosystem Econometric Analysis")
    print("=" * 50)
    
    # Initialize data fetcher
    fetcher = AmazonDataFetcher()
    
    # Fetch data from all available sources
    yahoo_success = fetcher.fetch_yahoo_finance_data()
    alpha_success = fetcher.fetch_alpha_vantage_data()
    quandl_success = fetcher.fetch_quandl_data()
    
    # Generate insights
    if yahoo_success:
        insights = fetcher.generate_amazon_business_insights()
        
        # Create a simple visualization
        if 'yahoo_finance' in fetcher.data:
            history = fetcher.data['yahoo_finance']['history']
            if not history.empty:
                plt.figure(figsize=(12, 6))
                plt.plot(history.index, history['Close'], linewidth=2)
                plt.title('Amazon (AMZN) Stock Price - Last 5 Years')
                plt.xlabel('Date')
                plt.ylabel('Price ($)')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save the plot
                plt.savefig('d:/Econometric/Ecosystems/amazon_stock_chart.png', dpi=300, bbox_inches='tight')
                print(f"\n📊 Stock chart saved to: amazon_stock_chart.png")
                plt.show()
    
    # API setup instructions
    print(f"\n⚙️  API Setup Status:")
    print(f"• Yahoo Finance: ✅ Working (no API key required)")
    print(f"• Alpha Vantage: {'✅ Ready' if alpha_success else '⚠️  Need API key'}")
    print(f"• Quandl: {'✅ Ready' if quandl_success else '⚠️  Need API key'}")
    
    if not alpha_success or not quandl_success:
        print(f"\n🔧 To get additional data sources:")
        print(f"1. Get free API keys:")
        print(f"   • Alpha Vantage: https://www.alphavantage.co/support/#api-key")
        print(f"   • Quandl: https://www.quandl.com/tools/api")
        print(f"2. Add them to your .env file:")
        print(f"   ALPHA_VANTAGE_API_KEY=your_key_here")
        print(f"   QUANDL_API_KEY=your_key_here")

if __name__ == "__main__":
    main()
