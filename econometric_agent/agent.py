"""
Econometric Agent - Main orchestrating class for economic data analysis and modeling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
import warnings
import os
from datetime import datetime, timedelta

from .data_fetcher import EconomicDataFetcher
from .data_validator import DataValidator
from .models import EconometricModels


class EconometricAgent:
    """
    Professional econometric agent for comprehensive economic data analysis and modeling.
    
    This agent provides an integrated workflow for:
    - Fetching real economic data from validated sources
    - Validating and cleaning data quality
    - Building sophisticated econometric models
    - Generating professional analysis reports
    """
    
    def __init__(self, fred_api_key: Optional[str] = None, 
                 missing_threshold: float = 0.3,
                 outlier_method: str = 'iqr',
                 random_state: int = 42):
        """
        Initialize the Econometric Agent.
        
        Args:
            fred_api_key: FRED API key for data access
            missing_threshold: Maximum proportion of missing values allowed
            outlier_method: Method for outlier detection ('iqr', 'zscore', 'modified_zscore')
            random_state: Random seed for reproducibility
        """
        self.data_fetcher = EconomicDataFetcher(fred_api_key)
        self.data_validator = DataValidator(missing_threshold, outlier_method)
        self.econometric_models = EconometricModels(random_state)
        
        self.datasets = {}
        self.analysis_history = []
        self.current_data = None
        
    def load_economic_indicators(self, indicators: Optional[List[str]] = None,
                                start_date: Optional[str] = None,
                                end_date: Optional[str] = None,
                                dataset_name: str = "main") -> pd.DataFrame:
        """
        Load economic indicators from FRED database.
        
        Args:
            indicators: List of FRED series IDs. If None, loads common indicators
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            dataset_name: Name for storing the dataset
            
        Returns:
            DataFrame with loaded economic data
        """
        try:
            if indicators is None:
                # Load common economic indicators
                data = self.data_fetcher.get_common_economic_indicators(start_date, end_date)
            else:
                # Load specific indicators
                data = self.data_fetcher.get_multiple_fred_series(indicators, start_date, end_date)
            
            # Store dataset
            self.datasets[dataset_name] = data
            self.current_data = data
            
            # Log action
            self._log_action(f"Loaded economic data: {dataset_name}", {
                'indicators': indicators or 'common_indicators',
                'start_date': start_date,
                'end_date': end_date,
                'shape': data.shape
            })
            
            print(f"✓ Successfully loaded {data.shape[0]} observations across {data.shape[1]} indicators")
            return data
            
        except Exception as e:
            print(f"✗ Failed to load economic data: {e}")
            raise
    
    def validate_data(self, dataset_name: str = "main") -> Dict[str, Any]:
        """
        Validate data quality for the specified dataset.
        
        Args:
            dataset_name: Name of the dataset to validate
            
        Returns:
            Dictionary with validation results
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        data = self.datasets[dataset_name]
        results = self.data_validator.validate_data_quality(data)
        
        # Log action
        self._log_action(f"Validated data quality: {dataset_name}", results)
        
        # Print summary
        quality_score = results['quality_score']
        print(f"Data Quality Score: {quality_score:.1f}/100")
        
        if quality_score >= 80:
            print("✓ Data quality is excellent")
        elif quality_score >= 60:
            print("⚠ Data quality is acceptable but could be improved")
        else:
            print("✗ Data quality issues detected - review recommendations")
        
        return results
    
    def clean_data(self, dataset_name: str = "main",
                   remove_outliers: bool = False,
                   interpolate_missing: bool = True,
                   method: str = 'linear') -> pd.DataFrame:
        """
        Clean the specified dataset.
        
        Args:
            dataset_name: Name of the dataset to clean
            remove_outliers: Whether to remove detected outliers
            interpolate_missing: Whether to interpolate missing values
            method: Interpolation method
            
        Returns:
            Cleaned DataFrame
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        data = self.datasets[dataset_name]
        cleaned_data = self.data_validator.clean_data(
            data, remove_outliers, interpolate_missing, method
        )
        
        # Store cleaned data
        cleaned_name = f"{dataset_name}_cleaned"
        self.datasets[cleaned_name] = cleaned_data
        self.current_data = cleaned_data
        
        # Log action
        self._log_action(f"Cleaned data: {dataset_name}", {
            'remove_outliers': remove_outliers,
            'interpolate_missing': interpolate_missing,
            'method': method,
            'original_shape': data.shape,
            'cleaned_shape': cleaned_data.shape
        })
        
        print(f"✓ Data cleaned and saved as '{cleaned_name}'")
        return cleaned_data
    
    def build_regression_model(self, target_variable: str,
                              feature_variables: Optional[List[str]] = None,
                              dataset_name: str = "main",
                              model_type: str = 'linear',
                              **kwargs) -> Dict[str, Any]:
        """
        Build a regression model for economic forecasting.
        
        Args:
            target_variable: Name of the target variable
            feature_variables: List of feature variables. If None, uses all others
            dataset_name: Name of the dataset to use
            model_type: Type of regression ('linear', 'ridge', 'lasso', 'elastic_net')
            **kwargs: Additional arguments for the specific model
            
        Returns:
            Dictionary with model results
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        data = self.datasets[dataset_name]
        
        if target_variable not in data.columns:
            raise ValueError(f"Target variable '{target_variable}' not found in dataset")
        
        try:
            if model_type == 'linear':
                results = self.econometric_models.linear_regression(
                    data, target_variable, feature_variables, **kwargs
                )
            elif model_type in ['ridge', 'lasso', 'elastic_net']:
                results = self.econometric_models.regularized_regression(
                    data, target_variable, feature_variables, model_type, **kwargs
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Log action
            self._log_action(f"Built {model_type} regression model", {
                'target': target_variable,
                'features': feature_variables,
                'dataset': dataset_name,
                'performance': {
                    'r2_test': results.get('r2_test'),
                    'mse_test': results.get('mse_test')
                }
            })
            
            print(f"✓ {model_type.title()} regression model built successfully")
            print(f"  R² (Test): {results.get('r2_test', 'N/A'):.4f}")
            print(f"  MSE (Test): {results.get('mse_test', 'N/A'):.4f}")
            
            return results
            
        except Exception as e:
            print(f"✗ Failed to build regression model: {e}")
            raise
    
    def build_time_series_model(self, target_variable: str,
                               dataset_name: str = "main",
                               model_type: str = 'arima',
                               **kwargs) -> Dict[str, Any]:
        """
        Build a time series model for economic forecasting.
        
        Args:
            target_variable: Name of the target variable
            dataset_name: Name of the dataset to use
            model_type: Type of time series model ('arima', 'var')
            **kwargs: Additional arguments for the specific model
            
        Returns:
            Dictionary with model results
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        data = self.datasets[dataset_name]
        
        try:
            if model_type == 'arima':
                if target_variable not in data.columns:
                    raise ValueError(f"Target variable '{target_variable}' not found")
                
                series = data[target_variable]
                results = self.econometric_models.time_series_arima(series, **kwargs)
                
            elif model_type == 'var':
                # For VAR, use all variables or a subset
                if isinstance(target_variable, list):
                    var_data = data[target_variable]
                else:
                    var_data = data
                
                results = self.econometric_models.vector_autoregression(var_data, **kwargs)
                
            else:
                raise ValueError(f"Unknown time series model type: {model_type}")
            
            # Log action
            self._log_action(f"Built {model_type} time series model", {
                'target': target_variable,
                'dataset': dataset_name,
                'performance': {
                    'aic': results.get('aic'),
                    'bic': results.get('bic')
                }
            })
            
            print(f"✓ {model_type.upper()} time series model built successfully")
            print(f"  AIC: {results.get('aic', 'N/A'):.4f}")
            print(f"  BIC: {results.get('bic', 'N/A'):.4f}")
            
            return results
            
        except Exception as e:
            print(f"✗ Failed to build time series model: {e}")
            raise
    
    def analyze_stationarity(self, variable: str, dataset_name: str = "main") -> Dict[str, Any]:
        """
        Analyze stationarity of a time series variable.
        
        Args:
            variable: Name of the variable to test
            dataset_name: Name of the dataset to use
            
        Returns:
            Dictionary with stationarity test results
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        data = self.datasets[dataset_name]
        
        if variable not in data.columns:
            raise ValueError(f"Variable '{variable}' not found in dataset")
        
        series = data[variable]
        results = self.econometric_models.test_stationarity(series)
        
        # Log action
        self._log_action(f"Analyzed stationarity: {variable}", results)
        
        print(f"Stationarity Analysis for {variable}:")
        print(f"  {results['interpretation']}")
        print(f"  ADF Test p-value: {results['adf_test']['p_value']:.4f}")
        print(f"  KPSS Test p-value: {results['kpss_test']['p_value']:.4f}")
        
        return results
    
    def generate_forecast(self, model_name: str, steps: int = 12) -> pd.DataFrame:
        """
        Generate forecasts using a fitted model.
        
        Args:
            model_name: Name of the fitted model
            steps: Number of steps to forecast
            
        Returns:
            DataFrame with forecasts
        """
        if model_name not in self.econometric_models.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model_info = self.econometric_models.models[model_name]
        
        try:
            if 'arima' in model_name.lower():
                fitted_model = model_info['fitted_model']
                forecast = fitted_model.forecast(steps=steps)
                forecast_ci = fitted_model.get_forecast(steps=steps).conf_int()
                
                # Create forecast DataFrame
                forecast_df = pd.DataFrame({
                    'forecast': forecast,
                    'lower_ci': forecast_ci.iloc[:, 0],
                    'upper_ci': forecast_ci.iloc[:, 1]
                })
                
            elif 'var' in model_name.lower():
                fitted_model = model_info['fitted_model']
                last_obs = self.current_data.iloc[-model_info['optimal_lags']:].values
                forecast = fitted_model.forecast(last_obs, steps=steps)
                
                forecast_df = pd.DataFrame(forecast, columns=model_info['variables'])
                
            else:
                raise ValueError(f"Forecasting not implemented for model type: {model_name}")
            
            # Log action
            self._log_action(f"Generated forecast: {model_name}", {
                'steps': steps,
                'forecast_shape': forecast_df.shape
            })
            
            print(f"✓ Generated {steps}-step forecast using {model_name}")
            return forecast_df
            
        except Exception as e:
            print(f"✗ Failed to generate forecast: {e}")
            raise
    
    def compare_models(self, models_to_compare: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare performance of multiple models.
        
        Args:
            models_to_compare: List of model names. If None, compares all
            
        Returns:
            DataFrame with model comparison
        """
        comparison = self.econometric_models.model_comparison(models_to_compare)
        
        if not comparison.empty:
            print("Model Comparison:")
            print("=" * 50)
            print(comparison.to_string(index=False))
        
        return comparison
    
    def generate_report(self, model_name: Optional[str] = None,
                       dataset_name: str = "main") -> str:
        """
        Generate a comprehensive analysis report.
        
        Args:
            model_name: Specific model to report on. If None, generates summary
            dataset_name: Dataset to include in report
            
        Returns:
            Formatted report string
        """
        if model_name:
            # Generate model-specific report
            return self.econometric_models.generate_model_report(model_name)
        else:
            # Generate comprehensive report
            report = []
            report.append("=" * 80)
            report.append("ECONOMETRIC ANALYSIS REPORT")
            report.append("=" * 80)
            report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("")
            
            # Dataset summary
            if dataset_name in self.datasets:
                data = self.datasets[dataset_name]
                report.append(f"DATASET: {dataset_name}")
                report.append("-" * 40)
                report.append(f"Shape: {data.shape}")
                report.append(f"Variables: {', '.join(data.columns)}")
                report.append(f"Date Range: {data.index.min()} to {data.index.max()}")
                report.append("")
            
            # Model summary
            if self.econometric_models.results:
                report.append("MODELS FITTED:")
                report.append("-" * 40)
                for model_name in self.econometric_models.results:
                    report.append(f"• {model_name}")
                report.append("")
                
                # Model comparison
                comparison = self.econometric_models.model_comparison()
                if not comparison.empty:
                    report.append("MODEL COMPARISON:")
                    report.append("-" * 40)
                    report.append(comparison.to_string(index=False))
                    report.append("")
            
            # Analysis history
            if self.analysis_history:
                report.append("ANALYSIS HISTORY:")
                report.append("-" * 40)
                for i, action in enumerate(self.analysis_history[-10:], 1):  # Last 10 actions
                    report.append(f"{i}. {action['action']} ({action['timestamp']})")
            
            return "\n".join(report)
    
    def _log_action(self, action: str, details: Dict[str, Any]):
        """Log an analysis action for tracking."""
        self.analysis_history.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'action': action,
            'details': details
        })
    
    def get_data_info(self, dataset_name: str = "main") -> str:
        """
        Get information about a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Formatted information string
        """
        if dataset_name not in self.datasets:
            return f"Dataset '{dataset_name}' not found"
        
        data = self.datasets[dataset_name]
        
        info = []
        info.append(f"Dataset: {dataset_name}")
        info.append(f"Shape: {data.shape}")
        info.append(f"Variables: {', '.join(data.columns)}")
        info.append(f"Date Range: {data.index.min()} to {data.index.max()}")
        info.append(f"Missing Values: {data.isnull().sum().sum()}")
        
        return "\n".join(info)
    
    def list_available_indicators(self, search_term: Optional[str] = None) -> pd.DataFrame:
        """
        Search for available economic indicators from FRED.
        
        Args:
            search_term: Search term for finding indicators
            
        Returns:
            DataFrame with available indicators
        """
        if search_term:
            try:
                results = self.data_fetcher.search_fred_series(search_term)
                print(f"Found {len(results)} indicators matching '{search_term}'")
                return results
            except Exception as e:
                print(f"Search failed: {e}")
                return pd.DataFrame()
        else:
            # Return common indicators
            common_indicators = {
                'GDP': 'Gross Domestic Product',
                'UNRATE': 'Unemployment Rate',
                'CPIAUCSL': 'Consumer Price Index',
                'FEDFUNDS': 'Federal Funds Rate',
                'INDPRO': 'Industrial Production Index',
                'RSAFS': 'Retail Sales',
                'UMCSENT': 'Consumer Sentiment',
                'SP500': 'S&P 500'
            }
            
            df = pd.DataFrame([
                {'id': k, 'title': v} for k, v in common_indicators.items()
            ])
            
            print("Common Economic Indicators:")
            return df