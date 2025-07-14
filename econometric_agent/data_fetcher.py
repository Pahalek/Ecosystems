"""
Economic Data Fetcher - Retrieves real economic data from validated sources.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import requests
import os
from fredapi import Fred
import warnings


class EconomicDataFetcher:
    """
    Fetches economic data from multiple reliable sources including FRED, World Bank, etc.
    """
    
    def __init__(self, fred_api_key: Optional[str] = None):
        """
        Initialize the data fetcher.
        
        Args:
            fred_api_key: FRED API key. If None, will try to get from environment.
        """
        self.fred_api_key = fred_api_key or os.getenv('FRED_API_KEY')
        self.fred = None
        
        if self.fred_api_key:
            try:
                self.fred = Fred(api_key=self.fred_api_key)
            except Exception as e:
                warnings.warn(f"Failed to initialize FRED API: {e}")
    
    def get_fred_data(self, series_id: str, start_date: Optional[str] = None, 
                      end_date: Optional[str] = None) -> pd.Series:
        """
        Fetch data from FRED database.
        
        Args:
            series_id: FRED series identifier (e.g., 'GDP', 'UNRATE', 'CPIAUCSL')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            pandas Series with the requested data
        """
        if not self.fred:
            raise ValueError("FRED API not initialized. Please provide a valid API key.")
        
        try:
            data = self.fred.get_series(
                series_id,
                start=start_date,
                end=end_date
            )
            return data
        except Exception as e:
            raise ValueError(f"Failed to fetch FRED data for {series_id}: {e}")
    
    def get_multiple_fred_series(self, series_ids: List[str], 
                                start_date: Optional[str] = None,
                                end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch multiple FRED series and combine into a DataFrame.
        
        Args:
            series_ids: List of FRED series identifiers
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            pandas DataFrame with multiple series
        """
        data_dict = {}
        
        for series_id in series_ids:
            try:
                data_dict[series_id] = self.get_fred_data(series_id, start_date, end_date)
            except Exception as e:
                warnings.warn(f"Failed to fetch {series_id}: {e}")
                continue
        
        if not data_dict:
            raise ValueError("No data could be fetched for any of the requested series")
        
        df = pd.DataFrame(data_dict)
        return df
    
    def get_world_bank_data(self, indicator: str, country: str = "US", 
                           start_year: Optional[int] = None, 
                           end_year: Optional[int] = None) -> pd.Series:
        """
        Fetch data from World Bank API.
        
        Args:
            indicator: World Bank indicator code
            country: Country code (default: "US")
            start_year: Start year
            end_year: End year
            
        Returns:
            pandas Series with the requested data
        """
        base_url = "https://api.worldbank.org/v2/country"
        
        # Build URL
        url = f"{base_url}/{country}/indicator/{indicator}"
        params = {
            "format": "json",
            "per_page": "1000"
        }
        
        if start_year:
            params["date"] = f"{start_year}:{end_year or datetime.now().year}"
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if len(data) < 2 or not data[1]:
                raise ValueError(f"No data available for indicator {indicator}")
            
            # Convert to pandas Series
            records = data[1]
            dates = []
            values = []
            
            for record in records:
                if record['value'] is not None:
                    dates.append(pd.to_datetime(record['date']))
                    values.append(float(record['value']))
            
            series = pd.Series(values, index=dates, name=indicator)
            return series.sort_index()
            
        except Exception as e:
            raise ValueError(f"Failed to fetch World Bank data: {e}")
    
    def get_common_economic_indicators(self, start_date: Optional[str] = None,
                                     end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch commonly used economic indicators from FRED.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with common economic indicators
        """
        common_series = {
            'GDP': 'GDP',  # Gross Domestic Product
            'UNEMPLOYMENT': 'UNRATE',  # Unemployment Rate
            'INFLATION': 'CPIAUCSL',  # Consumer Price Index
            'FEDERAL_FUNDS_RATE': 'FEDFUNDS',  # Federal Funds Rate
            'INDUSTRIAL_PRODUCTION': 'INDPRO',  # Industrial Production Index
            'RETAIL_SALES': 'RSAFS',  # Retail Sales
            'CONSUMER_SENTIMENT': 'UMCSENT',  # Consumer Sentiment
            'SP500': 'SP500'  # S&P 500
        }
        
        data = self.get_multiple_fred_series(
            list(common_series.values()),
            start_date,
            end_date
        )
        
        # Rename columns to more descriptive names
        column_mapping = {v: k for k, v in common_series.items()}
        data = data.rename(columns=column_mapping)
        
        return data
    
    def search_fred_series(self, search_text: str, limit: int = 10) -> pd.DataFrame:
        """
        Search for FRED series based on text.
        
        Args:
            search_text: Text to search for
            limit: Maximum number of results
            
        Returns:
            DataFrame with search results
        """
        if not self.fred:
            raise ValueError("FRED API not initialized. Please provide a valid API key.")
        
        try:
            results = self.fred.search(search_text, limit=limit)
            return results
        except Exception as e:
            raise ValueError(f"Failed to search FRED: {e}")