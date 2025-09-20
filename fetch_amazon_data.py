"""
Fetches Amazon (AMZN) business and financial data from Yahoo Finance, Alpha Vantage, and Quandl.
Requires API keys for Alpha Vantage and Quandl (add to .env as ALPHA_VANTAGE_API_KEY and QUANDL_API_KEY).
"""
import os
from dotenv import load_dotenv
import yfinance as yf
from alpha_vantage.fundamentaldata import FundamentalData
import quandl

# Load API keys from .env
load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
QUANDL_API_KEY = os.getenv("QUANDL_API_KEY")

# Yahoo Finance: Fetch AMZN stock and financials
def fetch_yahoo_finance_amzn():
    amzn = yf.Ticker("AMZN")
    info = amzn.info
    hist = amzn.history(period="max")
    return {"info": info, "history": hist}

# Alpha Vantage: Fetch AMZN fundamentals
def fetch_alpha_vantage_amzn():
    if not ALPHA_VANTAGE_API_KEY:
        raise ValueError("Missing ALPHA_VANTAGE_API_KEY in .env")
    fd = FundamentalData(ALPHA_VANTAGE_API_KEY)
    balance_sheet, _ = fd.get_balance_sheet_annual("AMZN")
    income_statement, _ = fd.get_income_statement_annual("AMZN")
    cash_flow, _ = fd.get_cash_flow_annual("AMZN")
    return {
        "balance_sheet": balance_sheet,
        "income_statement": income_statement,
        "cash_flow": cash_flow
    }

# Quandl: Fetch Amazon-related macro/e-commerce data
def fetch_quandl_amzn():
    if not QUANDL_API_KEY:
        raise ValueError("Missing QUANDL_API_KEY in .env")
    quandl.ApiConfig.api_key = QUANDL_API_KEY
    # Example: NASDAQ/AMZN stock price (Quandl code may change)
    try:
        amzn_stock = quandl.get("NASDAQ/AMZN")
    except Exception as e:
        amzn_stock = str(e)
    return {"amzn_stock": amzn_stock}

if __name__ == "__main__":
    print("Fetching Yahoo Finance data...")
    ydata = fetch_yahoo_finance_amzn()
    print("Yahoo Finance info:", ydata["info"])
    print("Fetching Alpha Vantage data...")
    try:
        avdata = fetch_alpha_vantage_amzn()
        print("Alpha Vantage balance sheet:", avdata["balance_sheet"])
    except Exception as e:
        print("Alpha Vantage error:", e)
    print("Fetching Quandl data...")
    try:
        qdata = fetch_quandl_amzn()
        print("Quandl AMZN stock:", qdata["amzn_stock"])
    except Exception as e:
        print("Quandl error:", e)
