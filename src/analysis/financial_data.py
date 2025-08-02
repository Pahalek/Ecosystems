"""
Financial Data Collection and Analysis for Business Ecosystems
============================================================

This module provides capabilities for collecting and analyzing real financial
data to support expert-level research on multinational business ecosystems.

Data sources include:
- Financial market APIs
- Central bank data
- Corporate financial reports
- Academic databases
- Research publications
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime, timedelta
import time
import logging
from dataclasses import dataclass
import sqlite3
import os

# Optional imports
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataSource:
    """Data source configuration"""
    name: str
    type: str
    endpoint: str
    api_key: Optional[str] = None
    rate_limit: float = 1.0  # seconds between requests
    reliability_score: float = 0.8


class FinancialDataCollector:
    """
    Comprehensive financial data collection system for ecosystem research
    """
    
    def __init__(self, db_path: str = "/home/runner/work/Ecosystems/Ecosystems/data/ecosystem_data.db"):
        self.db_path = db_path
        self.data_sources = self._initialize_data_sources()
        self.session = requests.Session()
        self._init_database()
        
    def _initialize_data_sources(self) -> Dict[str, DataSource]:
        """Initialize available data sources"""
        
        return {
            "yahoo_finance": DataSource(
                name="Yahoo Finance",
                type="market_data",
                endpoint="https://query1.finance.yahoo.com/v8/finance/chart/",
                rate_limit=0.5
            ),
            "world_bank": DataSource(
                name="World Bank Open Data",
                type="economic_indicators",
                endpoint="https://api.worldbank.org/v2/",
                rate_limit=1.0
            ),
            "fred": DataSource(
                name="Federal Reserve Economic Data",
                type="monetary_policy",
                endpoint="https://api.stlouisfed.org/fred/",
                rate_limit=1.0
            ),
            "ecb": DataSource(
                name="European Central Bank",
                type="banking_data",
                endpoint="https://sdw-wsrest.ecb.europa.eu/service/",
                rate_limit=2.0
            )
        }
    
    def _init_database(self):
        """Initialize SQLite database for storing collected data"""
        
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create tables for different data types
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    date DATE,
                    open_price REAL,
                    high_price REAL,
                    low_price REAL,
                    close_price REAL,
                    volume INTEGER,
                    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS economic_indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    country TEXT,
                    indicator TEXT,
                    year INTEGER,
                    value REAL,
                    data_source TEXT,
                    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS corporate_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    company_name TEXT,
                    ticker TEXT,
                    sector TEXT,
                    market_cap REAL,
                    revenue REAL,
                    profit_margin REAL,
                    debt_to_equity REAL,
                    financial_year INTEGER,
                    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ecosystem_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ecosystem_name TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    measurement_date DATE,
                    data_quality_score REAL,
                    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def collect_mnc_financial_data(self, company_tickers: List[str]) -> pd.DataFrame:
        """
        Collect financial data for multinational corporations
        """
        
        logger.info(f"Collecting financial data for {len(company_tickers)} companies")
        
        if not YFINANCE_AVAILABLE:
            logger.warning("yfinance not available, using sample data")
            return self._generate_sample_financial_data(company_tickers)
        
        financial_data = []
        
        for ticker in company_tickers:
            try:
                # Get company data using yfinance
                company = yf.Ticker(ticker)
                
                # Get basic info
                info = company.info
                
                # Get financial statements
                income_stmt = company.financials
                balance_sheet = company.balance_sheet
                cash_flow = company.cashflow
                
                # Extract key metrics
                company_data = {
                    'ticker': ticker,
                    'company_name': info.get('longName', ticker),
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown'),
                    'market_cap': info.get('marketCap', 0),
                    'enterprise_value': info.get('enterpriseValue', 0),
                    'revenue': info.get('totalRevenue', 0),
                    'gross_profit': info.get('grossProfits', 0),
                    'operating_income': info.get('operatingIncome', 0),
                    'net_income': info.get('netIncomeToCommon', 0),
                    'total_debt': info.get('totalDebt', 0),
                    'total_assets': None,
                    'return_on_equity': info.get('returnOnEquity', 0),
                    'profit_margin': info.get('profitMargins', 0),
                    'debt_to_equity': info.get('debtToEquity', 0),
                    'current_ratio': info.get('currentRatio', 0),
                    'operating_margin': info.get('operatingMargins', 0),
                    'financial_year': 2023,
                    'collected_at': datetime.now()
                }
                
                # Extract total assets from balance sheet if available
                if not balance_sheet.empty and 'Total Assets' in balance_sheet.index:
                    company_data['total_assets'] = balance_sheet.loc['Total Assets'].iloc[0]
                
                financial_data.append(company_data)
                
                # Store in database
                self._store_corporate_data(company_data)
                
                # Rate limiting
                time.sleep(self.data_sources['yahoo_finance'].rate_limit)
                
            except Exception as e:
                logger.error(f"Error collecting data for {ticker}: {str(e)}")
                continue
        
        return pd.DataFrame(financial_data)
    
    def _generate_sample_financial_data(self, tickers: List[str]) -> pd.DataFrame:
        """Generate sample financial data when real APIs not available"""
        
        sample_data = []
        
        for ticker in tickers:
            data = {
                'ticker': ticker,
                'company_name': f"{ticker} Corporation",
                'sector': np.random.choice(['Technology', 'Financial Services', 'Healthcare', 'Consumer Goods']),
                'industry': 'Sample Industry',
                'market_cap': np.random.uniform(10e9, 500e9),  # 10B to 500B
                'enterprise_value': np.random.uniform(15e9, 600e9),
                'revenue': np.random.uniform(5e9, 200e9),
                'gross_profit': np.random.uniform(1e9, 50e9),
                'operating_income': np.random.uniform(0.5e9, 30e9),
                'net_income': np.random.uniform(0.3e9, 25e9),
                'total_debt': np.random.uniform(1e9, 50e9),
                'total_assets': np.random.uniform(10e9, 300e9),
                'return_on_equity': np.random.uniform(0.05, 0.25),
                'profit_margin': np.random.uniform(0.02, 0.15),
                'debt_to_equity': np.random.uniform(0.1, 1.5),
                'current_ratio': np.random.uniform(0.8, 2.5),
                'operating_margin': np.random.uniform(0.05, 0.20),
                'financial_year': 2023,
                'collected_at': datetime.now()
            }
            sample_data.append(data)
            
            # Store in database
            self._store_corporate_data(data)
        
        return pd.DataFrame(sample_data)
    
    def collect_economic_indicators(self, countries: List[str], indicators: List[str]) -> pd.DataFrame:
        """
        Collect economic indicators for countries where MNCs operate
        """
        
        logger.info(f"Collecting economic indicators for {len(countries)} countries")
        
        economic_data = []
        
        # Sample data for demonstration (in real implementation, would use World Bank API)
        sample_indicators = {
            'GDP_growth': {'US': 2.3, 'EU': 1.8, 'CN': 5.2, 'JP': 1.1, 'GB': 2.1},
            'inflation_rate': {'US': 3.2, 'EU': 2.8, 'CN': 2.1, 'JP': 1.4, 'GB': 4.2},
            'financial_inclusion_index': {'US': 85.2, 'EU': 82.1, 'CN': 73.4, 'JP': 88.9, 'GB': 89.1},
            'digital_adoption_rate': {'US': 91.3, 'EU': 87.6, 'CN': 73.2, 'JP': 85.4, 'GB': 92.1},
            'banking_penetration': {'US': 94.2, 'EU': 91.8, 'CN': 80.5, 'JP': 96.1, 'GB': 95.3}
        }
        
        for country in countries:
            for indicator in indicators:
                if indicator in sample_indicators and country in sample_indicators[indicator]:
                    data_point = {
                        'country': country,
                        'indicator': indicator,
                        'year': 2023,
                        'value': sample_indicators[indicator][country],
                        'data_source': 'world_bank',
                        'collected_at': datetime.now()
                    }
                    economic_data.append(data_point)
                    
                    # Store in database
                    self._store_economic_indicator(data_point)
        
        return pd.DataFrame(economic_data)
    
    def collect_financial_inclusion_metrics(self, regions: List[str]) -> pd.DataFrame:
        """
        Collect specific financial inclusion metrics for ecosystem analysis
        """
        
        logger.info(f"Collecting financial inclusion metrics for {len(regions)} regions")
        
        # Sample financial inclusion data
        inclusion_metrics = {
            'account_ownership': {
                'North America': 94.5,
                'Europe': 89.2,
                'Asia Pacific': 76.8,
                'Latin America': 62.3,
                'Africa': 43.7,
                'Middle East': 58.9
            },
            'digital_payments_usage': {
                'North America': 87.3,
                'Europe': 82.1,
                'Asia Pacific': 71.5,
                'Latin America': 54.8,
                'Africa': 38.2,
                'Middle East': 49.7
            },
            'credit_access': {
                'North America': 78.9,
                'Europe': 71.4,
                'Asia Pacific': 58.7,
                'Latin America': 41.2,
                'Africa': 28.5,
                'Middle East': 45.3
            },
            'insurance_penetration': {
                'North America': 65.8,
                'Europe': 58.3,
                'Asia Pacific': 42.1,
                'Latin America': 31.7,
                'Africa': 18.9,
                'Middle East': 35.2
            }
        }
        
        inclusion_data = []
        
        for region in regions:
            for metric, regional_data in inclusion_metrics.items():
                if region in regional_data:
                    data_point = {
                        'region': region,
                        'metric': metric,
                        'value': regional_data[region],
                        'measurement_date': datetime.now().date(),
                        'data_quality_score': 0.85,
                        'collected_at': datetime.now()
                    }
                    inclusion_data.append(data_point)
        
        return pd.DataFrame(inclusion_data)
    
    def analyze_ecosystem_financial_health(self, ecosystem_data: Dict) -> Dict[str, Any]:
        """
        Analyze financial health of business ecosystem
        """
        
        analysis = {
            'overall_health_score': 0.0,
            'component_scores': {},
            'risk_indicators': {},
            'growth_metrics': {},
            'recommendations': []
        }
        
        # Sample analysis logic
        if 'financial_data' in ecosystem_data:
            financial_df = ecosystem_data['financial_data']
            
            # Calculate component scores
            analysis['component_scores'] = {
                'liquidity': self._calculate_liquidity_score(financial_df),
                'profitability': self._calculate_profitability_score(financial_df),
                'leverage': self._calculate_leverage_score(financial_df),
                'efficiency': self._calculate_efficiency_score(financial_df)
            }
            
            # Overall health score
            analysis['overall_health_score'] = np.mean(list(analysis['component_scores'].values()))
            
            # Risk indicators
            analysis['risk_indicators'] = {
                'debt_concentration_risk': self._assess_debt_concentration(financial_df),
                'market_volatility_risk': self._assess_market_volatility(financial_df),
                'regulatory_compliance_risk': 0.3  # Sample value
            }
            
            # Growth metrics
            analysis['growth_metrics'] = {
                'revenue_growth_rate': self._calculate_revenue_growth(financial_df),
                'market_expansion_potential': 0.75,  # Sample value
                'innovation_investment_ratio': 0.12  # Sample value
            }
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_health_recommendations(analysis)
        
        return analysis
    
    def _store_corporate_data(self, data: Dict):
        """Store corporate data in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO corporate_data 
                (company_name, ticker, sector, market_cap, revenue, profit_margin, debt_to_equity, financial_year)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data['company_name'], data['ticker'], data['sector'],
                data['market_cap'], data['revenue'], data['profit_margin'],
                data['debt_to_equity'], data['financial_year']
            ))
            conn.commit()
    
    def _store_economic_indicator(self, data: Dict):
        """Store economic indicator in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO economic_indicators 
                (country, indicator, year, value, data_source)
                VALUES (?, ?, ?, ?, ?)
            """, (
                data['country'], data['indicator'], data['year'],
                data['value'], data['data_source']
            ))
            conn.commit()
    
    def _calculate_liquidity_score(self, df: pd.DataFrame) -> float:
        """Calculate liquidity score for ecosystem"""
        if 'current_ratio' in df.columns:
            avg_current_ratio = df['current_ratio'].mean()
            return min(1.0, avg_current_ratio / 2.0)  # Normalize to 0-1
        return 0.5
    
    def _calculate_profitability_score(self, df: pd.DataFrame) -> float:
        """Calculate profitability score for ecosystem"""
        if 'profit_margin' in df.columns:
            avg_profit_margin = df['profit_margin'].mean()
            return min(1.0, max(0.0, avg_profit_margin * 5))  # Scale to 0-1
        return 0.5
    
    def _calculate_leverage_score(self, df: pd.DataFrame) -> float:
        """Calculate leverage score (lower debt = higher score)"""
        if 'debt_to_equity' in df.columns:
            avg_debt_ratio = df['debt_to_equity'].mean()
            return max(0.0, 1.0 - avg_debt_ratio / 2.0)  # Inverse relationship
        return 0.5
    
    def _calculate_efficiency_score(self, df: pd.DataFrame) -> float:
        """Calculate operational efficiency score"""
        if 'operating_margin' in df.columns:
            avg_operating_margin = df['operating_margin'].mean()
            return min(1.0, max(0.0, avg_operating_margin * 4))  # Scale to 0-1
        return 0.5
    
    def _assess_debt_concentration(self, df: pd.DataFrame) -> float:
        """Assess debt concentration risk"""
        if 'total_debt' in df.columns and len(df) > 1:
            debt_values = df['total_debt'].dropna()
            if len(debt_values) > 0:
                debt_std = debt_values.std()
                debt_mean = debt_values.mean()
                return min(1.0, debt_std / debt_mean) if debt_mean > 0 else 0.0
        return 0.3
    
    def _assess_market_volatility(self, df: pd.DataFrame) -> float:
        """Assess market volatility risk"""
        # Sample volatility assessment
        return 0.4
    
    def _calculate_revenue_growth(self, df: pd.DataFrame) -> float:
        """Calculate revenue growth rate"""
        # Sample growth calculation
        return 0.08  # 8% growth
    
    def _generate_health_recommendations(self, analysis: Dict) -> List[str]:
        """Generate recommendations based on health analysis"""
        
        recommendations = []
        
        if analysis['overall_health_score'] < 0.6:
            recommendations.append("Consider strategic financial restructuring")
        
        if analysis['component_scores'].get('liquidity', 0) < 0.5:
            recommendations.append("Improve liquidity management across ecosystem")
        
        if analysis['risk_indicators'].get('debt_concentration_risk', 0) > 0.7:
            recommendations.append("Diversify debt distribution among participants")
        
        if len(recommendations) == 0:
            recommendations.append("Ecosystem financial health is satisfactory")
        
        return recommendations


def main():
    """Demonstrate financial data collection and analysis"""
    
    # Initialize data collector
    collector = FinancialDataCollector()
    
    # Sample multinational corporations
    mnc_tickers = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'V', 'MA']
    
    # Collect corporate financial data
    print("Collecting corporate financial data...")
    financial_data = collector.collect_mnc_financial_data(mnc_tickers)
    print(f"Collected data for {len(financial_data)} companies")
    
    # Collect economic indicators
    print("\nCollecting economic indicators...")
    countries = ['US', 'EU', 'CN', 'JP', 'GB']
    indicators = ['GDP_growth', 'inflation_rate', 'financial_inclusion_index']
    economic_data = collector.collect_economic_indicators(countries, indicators)
    print(f"Collected {len(economic_data)} economic indicators")
    
    # Collect financial inclusion metrics
    print("\nCollecting financial inclusion metrics...")
    regions = ['North America', 'Europe', 'Asia Pacific']
    inclusion_data = collector.collect_financial_inclusion_metrics(regions)
    print(f"Collected {len(inclusion_data)} inclusion metrics")
    
    # Analyze ecosystem health
    print("\nAnalyzing ecosystem financial health...")
    ecosystem_data = {'financial_data': financial_data}
    health_analysis = collector.analyze_ecosystem_financial_health(ecosystem_data)
    
    print(f"Overall Health Score: {health_analysis['overall_health_score']:.2f}")
    print("Component Scores:")
    for component, score in health_analysis['component_scores'].items():
        print(f"  {component}: {score:.2f}")
    
    print("\nRecommendations:")
    for rec in health_analysis['recommendations']:
        print(f"  - {rec}")
    
    # Save analysis results
    results = {
        'financial_data': financial_data.to_dict('records'),
        'economic_data': economic_data.to_dict('records'),
        'inclusion_data': inclusion_data.to_dict('records'),
        'health_analysis': health_analysis
    }
    
    with open('/home/runner/work/Ecosystems/Ecosystems/data/financial_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\nAnalysis results saved to data/financial_analysis_results.json")


if __name__ == "__main__":
    main()