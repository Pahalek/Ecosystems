"""
Econometric Models - Professional econometric modeling capabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import statsmodels.api as sm
import statsmodels.tsa.api as tsa
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt
import seaborn as sns


class EconometricModels:
    """
    Comprehensive econometric modeling toolkit for economic data analysis.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the econometric models class.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.results = {}
        
    def linear_regression(self, data: pd.DataFrame, target_col: str,
                         feature_cols: Optional[List[str]] = None,
                         test_size: float = 0.2) -> Dict[str, Any]:
        """
        Perform linear regression analysis.
        
        Args:
            data: Input DataFrame
            target_col: Name of target variable column
            feature_cols: List of feature column names. If None, uses all except target
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with model results and statistics
        """
        if feature_cols is None:
            feature_cols = [col for col in data.columns if col != target_col]
        
        # Prepare data
        X = data[feature_cols].dropna()
        y = data[target_col].loc[X.index]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Fit OLS model using statsmodels for comprehensive statistics
        X_train_sm = sm.add_constant(X_train)
        X_test_sm = sm.add_constant(X_test)
        
        model_sm = sm.OLS(y_train, X_train_sm).fit()
        
        # Also fit sklearn model for additional metrics
        model_sk = LinearRegression().fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model_sk.predict(X_train)
        y_pred_test = model_sk.predict(X_test)
        
        # Store models
        model_name = f"linear_regression_{target_col}"
        self.models[model_name] = {
            'statsmodels': model_sm,
            'sklearn': model_sk,
            'feature_cols': feature_cols,
            'target_col': target_col
        }
        
        # Calculate metrics
        results = {
            'model_name': model_name,
            'r2_train': r2_score(y_train, y_pred_train),
            'r2_test': r2_score(y_test, y_pred_test),
            'mse_train': mean_squared_error(y_train, y_pred_train),
            'mse_test': mean_squared_error(y_test, y_pred_test),
            'mae_train': mean_absolute_error(y_train, y_pred_train),
            'mae_test': mean_absolute_error(y_test, y_pred_test),
            'statsmodels_summary': model_sm.summary(),
            'coefficients': dict(zip(['const'] + feature_cols, model_sm.params)),
            'p_values': dict(zip(['const'] + feature_cols, model_sm.pvalues)),
            'confidence_intervals': model_sm.conf_int(),
            'feature_importance': dict(zip(feature_cols, np.abs(model_sk.coef_))),
            'n_observations': len(X_train),
            'n_features': len(feature_cols)
        }
        
        self.results[model_name] = results
        return results
    
    def time_series_arima(self, data: pd.Series, order: Tuple[int, int, int] = None,
                         seasonal_order: Tuple[int, int, int, int] = None,
                         test_size: float = 0.2) -> Dict[str, Any]:
        """
        Fit ARIMA or SARIMA model to time series data.
        
        Args:
            data: Time series data
            order: ARIMA order (p, d, q). If None, will auto-select
            seasonal_order: Seasonal ARIMA order (P, D, Q, s). If None, non-seasonal
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with model results and forecasts
        """
        # Ensure data is sorted by index
        data = data.sort_index().dropna()
        
        # Split data
        split_idx = int(len(data) * (1 - test_size))
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        
        # Auto-select order if not provided
        if order is None:
            order = self._auto_arima_order(train_data)
        
        # Fit model
        if seasonal_order is not None:
            model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
        else:
            model = ARIMA(train_data, order=order)
        
        fitted_model = model.fit()
        
        # Generate forecasts
        forecast_steps = len(test_data)
        forecast = fitted_model.forecast(steps=forecast_steps)
        forecast_ci = fitted_model.get_forecast(steps=forecast_steps).conf_int()
        
        # In-sample predictions
        in_sample_pred = fitted_model.fittedvalues
        
        # Store model
        model_name = f"arima_{data.name or 'series'}_{order}"
        self.models[model_name] = {
            'fitted_model': fitted_model,
            'order': order,
            'seasonal_order': seasonal_order
        }
        
        # Calculate metrics
        in_sample_mse = mean_squared_error(train_data, in_sample_pred)
        out_sample_mse = mean_squared_error(test_data, forecast)
        
        results = {
            'model_name': model_name,
            'order': order,
            'seasonal_order': seasonal_order,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic,
            'in_sample_mse': in_sample_mse,
            'out_sample_mse': out_sample_mse,
            'forecast': forecast,
            'forecast_ci': forecast_ci,
            'in_sample_predictions': in_sample_pred,
            'residuals': fitted_model.resid,
            'model_summary': fitted_model.summary(),
            'ljung_box_test': acorr_ljungbox(fitted_model.resid, lags=10, return_df=True)
        }
        
        self.results[model_name] = results
        return results
    
    def vector_autoregression(self, data: pd.DataFrame, maxlags: int = 5,
                             test_size: float = 0.2) -> Dict[str, Any]:
        """
        Fit Vector Autoregression (VAR) model for multivariate time series.
        
        Args:
            data: Multivariate time series DataFrame
            maxlags: Maximum number of lags to consider
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with VAR model results
        """
        # Prepare data
        data_clean = data.dropna()
        
        # Split data
        split_idx = int(len(data_clean) * (1 - test_size))
        train_data = data_clean.iloc[:split_idx]
        test_data = data_clean.iloc[split_idx:]
        
        # Fit VAR model
        model = VAR(train_data)
        
        # Select optimal lag order
        lag_order_results = model.select_order(maxlags=maxlags)
        optimal_lags = lag_order_results.selected_orders['aic']
        
        # Fit with optimal lags
        fitted_model = model.fit(optimal_lags)
        
        # Generate forecasts
        forecast_steps = len(test_data)
        forecast = fitted_model.forecast(train_data.values[-optimal_lags:], steps=forecast_steps)
        forecast_df = pd.DataFrame(forecast, columns=data.columns, 
                                 index=test_data.index)
        
        # Store model
        model_name = f"var_{optimal_lags}lags"
        self.models[model_name] = {
            'fitted_model': fitted_model,
            'optimal_lags': optimal_lags,
            'variables': list(data.columns)
        }
        
        # Calculate metrics for each variable
        mse_results = {}
        for col in data.columns:
            mse_results[col] = mean_squared_error(test_data[col], forecast_df[col])
        
        results = {
            'model_name': model_name,
            'optimal_lags': optimal_lags,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic,
            'variables': list(data.columns),
            'forecast': forecast_df,
            'mse_by_variable': mse_results,
            'model_summary': fitted_model.summary(),
            'granger_causality': self._granger_causality_tests(fitted_model),
            'impulse_response': fitted_model.irf(periods=10)
        }
        
        self.results[model_name] = results
        return results
    
    def regularized_regression(self, data: pd.DataFrame, target_col: str,
                              feature_cols: Optional[List[str]] = None,
                              method: str = 'ridge', alpha: float = 1.0,
                              test_size: float = 0.2) -> Dict[str, Any]:
        """
        Perform regularized regression (Ridge, Lasso, or ElasticNet).
        
        Args:
            data: Input DataFrame
            target_col: Name of target variable column
            feature_cols: List of feature column names
            method: Regularization method ('ridge', 'lasso', 'elastic_net')
            alpha: Regularization strength
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with model results
        """
        if feature_cols is None:
            feature_cols = [col for col in data.columns if col != target_col]
        
        # Prepare data
        X = data[feature_cols].dropna()
        y = data[target_col].loc[X.index]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Select model
        if method == 'ridge':
            model = Ridge(alpha=alpha, random_state=self.random_state)
        elif method == 'lasso':
            model = Lasso(alpha=alpha, random_state=self.random_state)
        elif method == 'elastic_net':
            model = ElasticNet(alpha=alpha, random_state=self.random_state)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Store model
        model_name = f"{method}_{target_col}_alpha{alpha}"
        self.models[model_name] = {
            'model': model,
            'feature_cols': feature_cols,
            'target_col': target_col,
            'alpha': alpha
        }
        
        # Calculate metrics
        results = {
            'model_name': model_name,
            'method': method,
            'alpha': alpha,
            'r2_train': r2_score(y_train, y_pred_train),
            'r2_test': r2_score(y_test, y_pred_test),
            'mse_train': mean_squared_error(y_train, y_pred_train),
            'mse_test': mean_squared_error(y_test, y_pred_test),
            'mae_train': mean_absolute_error(y_train, y_pred_train),
            'mae_test': mean_absolute_error(y_test, y_pred_test),
            'coefficients': dict(zip(feature_cols, model.coef_)),
            'intercept': model.intercept_,
            'feature_importance': dict(zip(feature_cols, np.abs(model.coef_))),
            'n_observations': len(X_train),
            'n_features': len(feature_cols),
            'n_zero_coef': np.sum(np.abs(model.coef_) < 1e-6)
        }
        
        self.results[model_name] = results
        return results
    
    def _auto_arima_order(self, data: pd.Series, max_p: int = 3, max_d: int = 2, 
                         max_q: int = 3) -> Tuple[int, int, int]:
        """
        Automatically select ARIMA order using AIC criterion.
        """
        best_aic = np.inf
        best_order = (0, 0, 0)
        
        # Test stationarity to determine d
        d = 0
        series = data.copy()
        while d <= max_d:
            adf_result = adfuller(series.dropna())
            if adf_result[1] <= 0.05:  # p-value <= 0.05, reject null (series is stationary)
                break
            series = series.diff().dropna()
            d += 1
        
        # Grid search for p and q
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(data, order=(p, d, q))
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                except:
                    continue
        
        return best_order
    
    def _granger_causality_tests(self, var_model) -> Dict[str, Any]:
        """
        Perform Granger causality tests for VAR model.
        """
        causality_results = {}
        variables = var_model.names
        
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i != j:
                    try:
                        test_result = var_model.test_causality(var2, var1, kind='f')
                        causality_results[f"{var1}_causes_{var2}"] = {
                            'test_statistic': test_result.test_statistic,
                            'p_value': test_result.pvalue,
                            'significant': test_result.pvalue < 0.05
                        }
                    except:
                        continue
        
        return causality_results
    
    def test_stationarity(self, data: pd.Series) -> Dict[str, Any]:
        """
        Test time series stationarity using ADF and KPSS tests.
        
        Args:
            data: Time series data
            
        Returns:
            Dictionary with test results
        """
        # Augmented Dickey-Fuller test
        adf_result = adfuller(data.dropna())
        
        # KPSS test
        kpss_result = kpss(data.dropna())
        
        results = {
            'adf_test': {
                'statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4],
                'stationary': adf_result[1] <= 0.05
            },
            'kpss_test': {
                'statistic': kpss_result[0],
                'p_value': kpss_result[1],
                'critical_values': kpss_result[3],
                'stationary': kpss_result[1] > 0.05
            }
        }
        
        # Interpretation
        if results['adf_test']['stationary'] and results['kpss_test']['stationary']:
            results['interpretation'] = "Series is stationary"
        elif not results['adf_test']['stationary'] and not results['kpss_test']['stationary']:
            results['interpretation'] = "Series is non-stationary"
        else:
            results['interpretation'] = "Mixed results - further investigation needed"
        
        return results
    
    def model_comparison(self, models_to_compare: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare performance of multiple models.
        
        Args:
            models_to_compare: List of model names to compare. If None, compares all
            
        Returns:
            DataFrame with model comparison metrics
        """
        if models_to_compare is None:
            models_to_compare = list(self.results.keys())
        
        comparison_data = []
        
        for model_name in models_to_compare:
            if model_name in self.results:
                result = self.results[model_name]
                
                # Extract common metrics
                row = {'Model': model_name}
                
                if 'r2_test' in result:
                    row['R²_Test'] = result['r2_test']
                if 'mse_test' in result:
                    row['MSE_Test'] = result['mse_test']
                if 'mae_test' in result:
                    row['MAE_Test'] = result['mae_test']
                if 'aic' in result:
                    row['AIC'] = result['aic']
                if 'bic' in result:
                    row['BIC'] = result['bic']
                if 'n_observations' in result:
                    row['N_Obs'] = result['n_observations']
                
                comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def generate_model_report(self, model_name: str) -> str:
        """
        Generate a comprehensive report for a specific model.
        
        Args:
            model_name: Name of the model to report on
            
        Returns:
            Formatted model report string
        """
        if model_name not in self.results:
            return f"Model '{model_name}' not found."
        
        result = self.results[model_name]
        
        report = []
        report.append("=" * 60)
        report.append(f"MODEL REPORT: {model_name}")
        report.append("=" * 60)
        
        # Basic information
        if 'n_observations' in result:
            report.append(f"Observations: {result['n_observations']}")
        if 'n_features' in result:
            report.append(f"Features: {result['n_features']}")
        
        # Performance metrics
        report.append("\nPERFORMANCE METRICS:")
        report.append("-" * 30)
        
        if 'r2_test' in result:
            report.append(f"R² (Test): {result['r2_test']:.4f}")
        if 'mse_test' in result:
            report.append(f"MSE (Test): {result['mse_test']:.4f}")
        if 'mae_test' in result:
            report.append(f"MAE (Test): {result['mae_test']:.4f}")
        if 'aic' in result:
            report.append(f"AIC: {result['aic']:.4f}")
        if 'bic' in result:
            report.append(f"BIC: {result['bic']:.4f}")
        
        # Feature importance (if available)
        if 'feature_importance' in result:
            report.append("\nFEATURE IMPORTANCE:")
            report.append("-" * 30)
            sorted_features = sorted(result['feature_importance'].items(), 
                                   key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features[:10]:  # Top 10
                report.append(f"{feature}: {importance:.4f}")
        
        return "\n".join(report)