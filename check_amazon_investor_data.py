"""
Amazon Investor Data Analysis
Check what specific Amazon investor report data is accessible through our APIs.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
import yfinance as yf
from alpha_vantage.fundamentaldata import FundamentalData
import pandas as pd
from datetime import datetime

# Load environment variables
load_dotenv()

def check_amazon_investor_data():
    """Check comprehensive Amazon investor data availability."""
    print("🔍 AMAZON INVESTOR DATA ACCESSIBILITY CHECK")
    print("=" * 60)
    
    # Yahoo Finance comprehensive data check
    print("\n📊 YAHOO FINANCE DATA:")
    amzn = yf.Ticker("AMZN")
    
    # Basic info and recent data
    info = amzn.info
    print(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Data Source: Yahoo Finance")
    print(f"Most Recent Quarter: {info.get('mostRecentQuarter', 'N/A')}")
    if info.get('mostRecentQuarter'):
        most_recent = datetime.fromtimestamp(info.get('mostRecentQuarter'))
        print(f"Most Recent Quarter Date: {most_recent.strftime('%Y-%m-%d')}")
    
    # Key financial metrics from latest reports
    print(f"\n💰 LATEST FINANCIAL METRICS:")
    print(f"Total Revenue (TTM): ${info.get('totalRevenue', 0):,.0f}")
    print(f"Gross Profit: ${info.get('grossProfits', 0):,.0f}")
    print(f"Operating Cash Flow: ${info.get('operatingCashflow', 0):,.0f}")
    print(f"Free Cash Flow: ${info.get('freeCashflow', 0):,.0f}")
    print(f"Total Cash: ${info.get('totalCash', 0):,.0f}")
    print(f"Total Debt: ${info.get('totalDebt', 0):,.0f}")
    print(f"EBITDA: ${info.get('ebitda', 0):,.0f}")
    
    # Business segment info
    print(f"\n🏢 BUSINESS SEGMENTS:")
    business_summary = info.get('longBusinessSummary', '')
    if 'AWS' in business_summary:
        print("✅ AWS (Amazon Web Services) - mentioned")
    if 'North America' in business_summary:
        print("✅ North America segment - mentioned")
    if 'International' in business_summary:
        print("✅ International segment - mentioned")
    if 'advertising' in business_summary.lower():
        print("✅ Advertising services - mentioned")
    
    # Get financial statements
    print(f"\n📈 FINANCIAL STATEMENTS AVAILABILITY:")
    try:
        financials = amzn.financials
        balance_sheet = amzn.balance_sheet
        cashflow = amzn.cashflow
        
        print(f"✅ Income Statement: {len(financials.columns)} periods available")
        print(f"   Latest period: {financials.columns[0] if not financials.empty else 'N/A'}")
        
        print(f"✅ Balance Sheet: {len(balance_sheet.columns)} periods available")
        print(f"   Latest period: {balance_sheet.columns[0] if not balance_sheet.empty else 'N/A'}")
        
        print(f"✅ Cash Flow: {len(cashflow.columns)} periods available")
        print(f"   Latest period: {cashflow.columns[0] if not cashflow.empty else 'N/A'}")
        
        # Show recent revenue growth
        if not financials.empty and 'Total Revenue' in financials.index:
            recent_revenues = financials.loc['Total Revenue'].dropna()
            if len(recent_revenues) >= 2:
                growth = (recent_revenues.iloc[0] / recent_revenues.iloc[1] - 1) * 100
                print(f"   Recent Revenue Growth: {growth:.2f}%")
                
    except Exception as e:
        print(f"❌ Error accessing financial statements: {e}")
    
    # Alpha Vantage detailed fundamentals
    print(f"\n📊 ALPHA VANTAGE FUNDAMENTALS:")
    alpha_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if alpha_key:
        try:
            fd = FundamentalData(alpha_key)
            
            # Get annual reports
            balance_sheet, _ = fd.get_balance_sheet_annual("AMZN")
            income_statement, _ = fd.get_income_statement_annual("AMZN")
            cash_flow, _ = fd.get_cash_flow_annual("AMZN")
            
            print(f"✅ Annual Balance Sheet: {len(balance_sheet)} years of data")
            print(f"   Latest: {balance_sheet.iloc[0]['fiscalDateEnding'] if not balance_sheet.empty else 'N/A'}")
            
            print(f"✅ Annual Income Statement: {len(income_statement)} years of data")
            print(f"   Latest: {income_statement.iloc[0]['fiscalDateEnding'] if not income_statement.empty else 'N/A'}")
            
            print(f"✅ Annual Cash Flow: {len(cash_flow)} years of data")
            print(f"   Latest: {cash_flow.iloc[0]['fiscalDateEnding'] if not cash_flow.empty else 'N/A'}")
            
            # Show specific metrics from latest annual report
            if not income_statement.empty:
                latest = income_statement.iloc[0]
                print(f"\n💼 LATEST ANNUAL REPORT METRICS (Alpha Vantage):")
                print(f"Fiscal Year Ending: {latest.get('fiscalDateEnding', 'N/A')}")
                print(f"Total Revenue: ${float(latest.get('totalRevenue', 0)):,.0f}")
                print(f"Gross Profit: ${float(latest.get('grossProfit', 0)):,.0f}")
                print(f"Operating Income: ${float(latest.get('operatingIncome', 0)):,.0f}")
                print(f"Net Income: ${float(latest.get('netIncome', 0)):,.0f}")
                
            # Get quarterly data
            quarterly_income, _ = fd.get_income_statement_quarterly("AMZN")
            print(f"\n📅 QUARTERLY DATA:")
            print(f"✅ Quarterly Income Statements: {len(quarterly_income)} quarters available")
            if not quarterly_income.empty:
                latest_q = quarterly_income.iloc[0]
                print(f"   Latest Quarter: {latest_q.get('fiscalDateEnding', 'N/A')}")
                print(f"   Quarterly Revenue: ${float(latest_q.get('totalRevenue', 0)):,.0f}")
                
        except Exception as e:
            print(f"❌ Error accessing Alpha Vantage data: {e}")
    else:
        print("❌ Alpha Vantage API key not found")
    
    # Check earnings data
    print(f"\n📈 EARNINGS & ANALYST DATA:")
    try:
        # Analyst recommendations
        recommendations = amzn.recommendations
        if recommendations is not None and not recommendations.empty:
            print(f"✅ Analyst Recommendations: {len(recommendations)} recent recommendations")
            latest_rec = recommendations.iloc[-1] if not recommendations.empty else None
            if latest_rec is not None:
                print(f"   Latest: {latest_rec.name} - {latest_rec.get('To Grade', 'N/A')}")
        
        # Earnings estimates
        print(f"Current EPS (TTM): ${info.get('trailingEps', 0):.2f}")
        print(f"Forward EPS: ${info.get('forwardEps', 0):.2f}")
        print(f"Earnings Growth: {info.get('earningsGrowth', 0)*100:.1f}%")
        
    except Exception as e:
        print(f"⚠️  Earnings data: {e}")
    
    # Data recency assessment
    print(f"\n⏰ DATA RECENCY ASSESSMENT:")
    print("Yahoo Finance: Real-time/Daily updates")
    print("Alpha Vantage: Updated after earnings releases (quarterly/annual)")
    
    # What's missing from investor reports
    print(f"\n❓ WHAT MIGHT BE MISSING FROM LATEST INVESTOR REPORTS:")
    print("• AWS revenue breakdown (segment reporting)")
    print("• International vs North America performance details")
    print("• Prime membership numbers")
    print("• Advertising revenue specifics")
    print("• Forward-looking guidance statements")
    print("• Management commentary and outlook")
    
    print(f"\n💡 RECOMMENDATION:")
    print("For complete investor report data, consider:")
    print("• Amazon IR website: https://ir.aboutamazon.com/")
    print("• SEC EDGAR filings (10-K, 10-Q)")
    print("• Earnings call transcripts")
    print("• Current APIs provide comprehensive financial fundamentals")

if __name__ == "__main__":
    check_amazon_investor_data()
