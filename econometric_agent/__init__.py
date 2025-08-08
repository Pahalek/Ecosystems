"""
Econometric Agent - A professional economic data analysis and modeling system.

This package provides tools for fetching real economic data from reliable sources,
validating data quality, and building sophisticated econometric models.
"""

__version__ = "0.1.0"
__author__ = "Econometric Agent Team"

from .agent import EconometricAgent
from .data_fetcher import EconomicDataFetcher
from .data_validator import DataValidator
from .models import EconometricModels

__all__ = [
    "EconometricAgent",
    "EconomicDataFetcher", 
    "DataValidator",
    "EconometricModels"
]