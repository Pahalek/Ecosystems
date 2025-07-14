"""
Data Validator - Validates and cleans economic data for modeling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings
from scipy import stats


class DataValidator:
    """
    Validates and cleans economic data to ensure quality for econometric modeling.
    """
    
    def __init__(self, missing_threshold: float = 0.3, outlier_method: str = 'iqr'):
        """
        Initialize the data validator.
        
        Args:
            missing_threshold: Maximum proportion of missing values allowed (0.0 to 1.0)
            outlier_method: Method for outlier detection ('iqr', 'zscore', 'modified_zscore')
        """
        self.missing_threshold = missing_threshold
        self.outlier_method = outlier_method
    
    def validate_data_quality(self, data: Union[pd.DataFrame, pd.Series]) -> Dict[str, any]:
        """
        Comprehensive data quality assessment.
        
        Args:
            data: DataFrame or Series to validate
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'total_observations': len(data),
            'missing_data': {},
            'outliers': {},
            'data_types': {},
            'temporal_coverage': {},
            'quality_score': 0.0,
            'recommendations': []
        }
        
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        # Check missing data
        results['missing_data'] = self._check_missing_data(data)
        
        # Check outliers
        results['outliers'] = self._detect_outliers(data)
        
        # Check data types
        results['data_types'] = self._check_data_types(data)
        
        # Check temporal coverage
        if isinstance(data.index, pd.DatetimeIndex):
            results['temporal_coverage'] = self._check_temporal_coverage(data)
        
        # Calculate overall quality score
        results['quality_score'] = self._calculate_quality_score(results)
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results)
        
        return results
    
    def _check_missing_data(self, data: pd.DataFrame) -> Dict[str, any]:
        """Check for missing data patterns."""
        missing_info = {}
        
        for col in data.columns:
            missing_count = data[col].isnull().sum()
            missing_pct = missing_count / len(data)
            
            missing_info[col] = {
                'count': missing_count,
                'percentage': missing_pct,
                'acceptable': missing_pct <= self.missing_threshold
            }
        
        return missing_info
    
    def _detect_outliers(self, data: pd.DataFrame) -> Dict[str, any]:
        """Detect outliers using specified method."""
        outlier_info = {}
        
        for col in data.select_dtypes(include=[np.number]).columns:
            series = data[col].dropna()
            
            if self.outlier_method == 'iqr':
                outliers = self._detect_outliers_iqr(series)
            elif self.outlier_method == 'zscore':
                outliers = self._detect_outliers_zscore(series)
            elif self.outlier_method == 'modified_zscore':
                outliers = self._detect_outliers_modified_zscore(series)
            else:
                outliers = []
            
            outlier_info[col] = {
                'count': len(outliers),
                'percentage': len(outliers) / len(series) if len(series) > 0 else 0,
                'indices': outliers
            }
        
        return outlier_info
    
    def _detect_outliers_iqr(self, series: pd.Series) -> List[int]:
        """Detect outliers using Interquartile Range method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = series[(series < lower_bound) | (series > upper_bound)].index.tolist()
        return outliers
    
    def _detect_outliers_zscore(self, series: pd.Series, threshold: float = 3.0) -> List[int]:
        """Detect outliers using Z-score method."""
        z_scores = np.abs(stats.zscore(series))
        outliers = series[z_scores > threshold].index.tolist()
        return outliers
    
    def _detect_outliers_modified_zscore(self, series: pd.Series, threshold: float = 3.5) -> List[int]:
        """Detect outliers using Modified Z-score method."""
        median = np.median(series)
        mad = np.median(np.abs(series - median))
        modified_z_scores = 0.6745 * (series - median) / mad
        outliers = series[np.abs(modified_z_scores) > threshold].index.tolist()
        return outliers
    
    def _check_data_types(self, data: pd.DataFrame) -> Dict[str, str]:
        """Check data types and their appropriateness."""
        type_info = {}
        
        for col in data.columns:
            dtype = str(data[col].dtype)
            type_info[col] = dtype
        
        return type_info
    
    def _check_temporal_coverage(self, data: pd.DataFrame) -> Dict[str, any]:
        """Check temporal coverage and frequency."""
        temporal_info = {
            'start_date': data.index.min(),
            'end_date': data.index.max(),
            'frequency': pd.infer_freq(data.index),
            'gaps': [],
            'irregular_spacing': False
        }
        
        # Check for gaps in time series
        if temporal_info['frequency']:
            expected_index = pd.date_range(
                start=temporal_info['start_date'],
                end=temporal_info['end_date'],
                freq=temporal_info['frequency']
            )
            
            missing_dates = expected_index.difference(data.index)
            temporal_info['gaps'] = missing_dates.tolist()
        
        # Check for irregular spacing
        if len(data.index) > 1:
            date_diffs = data.index.to_series().diff().dropna()
            if date_diffs.nunique() > 1:
                temporal_info['irregular_spacing'] = True
        
        return temporal_info
    
    def _calculate_quality_score(self, results: Dict[str, any]) -> float:
        """Calculate an overall data quality score (0-100)."""
        score = 100.0
        
        # Penalize for missing data
        missing_penalty = 0
        for col_info in results['missing_data'].values():
            if not col_info['acceptable']:
                missing_penalty += col_info['percentage'] * 20
        score -= missing_penalty
        
        # Penalize for excessive outliers
        outlier_penalty = 0
        for col_info in results['outliers'].values():
            if col_info['percentage'] > 0.05:  # More than 5% outliers
                outlier_penalty += (col_info['percentage'] - 0.05) * 100
        score -= outlier_penalty
        
        # Penalize for temporal issues
        if 'temporal_coverage' in results and results['temporal_coverage']:
            if results['temporal_coverage']['gaps']:
                score -= len(results['temporal_coverage']['gaps']) * 2
            if results['temporal_coverage']['irregular_spacing']:
                score -= 10
        
        return max(0.0, min(100.0, score))
    
    def _generate_recommendations(self, results: Dict[str, any]) -> List[str]:
        """Generate data quality recommendations."""
        recommendations = []
        
        # Missing data recommendations
        for col, info in results['missing_data'].items():
            if not info['acceptable']:
                recommendations.append(
                    f"Column '{col}' has {info['percentage']:.1%} missing values. "
                    f"Consider imputation or removing this variable."
                )
        
        # Outlier recommendations
        for col, info in results['outliers'].items():
            if info['percentage'] > 0.05:
                recommendations.append(
                    f"Column '{col}' has {info['percentage']:.1%} outliers. "
                    f"Review for data errors or consider robust modeling methods."
                )
        
        # Temporal recommendations
        if 'temporal_coverage' in results and results['temporal_coverage']:
            if results['temporal_coverage']['gaps']:
                recommendations.append(
                    f"Found {len(results['temporal_coverage']['gaps'])} gaps in time series. "
                    f"Consider interpolation or acknowledge breaks in analysis."
                )
        
        return recommendations
    
    def clean_data(self, data: Union[pd.DataFrame, pd.Series], 
                   remove_outliers: bool = False,
                   interpolate_missing: bool = True,
                   method: str = 'linear') -> Union[pd.DataFrame, pd.Series]:
        """
        Clean the data based on validation results.
        
        Args:
            data: Data to clean
            remove_outliers: Whether to remove detected outliers
            interpolate_missing: Whether to interpolate missing values
            method: Interpolation method ('linear', 'polynomial', 'spline')
            
        Returns:
            Cleaned data
        """
        was_series = isinstance(data, pd.Series)
        if was_series:
            data = data.to_frame()
        
        cleaned_data = data.copy()
        
        # Handle missing values
        if interpolate_missing:
            for col in cleaned_data.select_dtypes(include=[np.number]).columns:
                if cleaned_data[col].isnull().any():
                    cleaned_data[col] = cleaned_data[col].interpolate(method=method)
        
        # Handle outliers
        if remove_outliers:
            outlier_results = self._detect_outliers(cleaned_data)
            for col, info in outlier_results.items():
                if info['indices']:
                    # Replace outliers with interpolated values
                    outlier_mask = cleaned_data.index.isin(info['indices'])
                    cleaned_data.loc[outlier_mask, col] = np.nan
                    cleaned_data[col] = cleaned_data[col].interpolate(method=method)
        
        return cleaned_data.iloc[:, 0] if was_series else cleaned_data
    
    def generate_quality_report(self, data: Union[pd.DataFrame, pd.Series]) -> str:
        """
        Generate a comprehensive data quality report.
        
        Args:
            data: Data to analyze
            
        Returns:
            Formatted quality report string
        """
        results = self.validate_data_quality(data)
        
        report = []
        report.append("=" * 60)
        report.append("DATA QUALITY REPORT")
        report.append("=" * 60)
        report.append(f"Total Observations: {results['total_observations']}")
        report.append(f"Quality Score: {results['quality_score']:.1f}/100")
        report.append("")
        
        # Missing data section
        report.append("MISSING DATA ANALYSIS:")
        report.append("-" * 30)
        for col, info in results['missing_data'].items():
            status = "✓" if info['acceptable'] else "✗"
            report.append(f"{status} {col}: {info['count']} missing ({info['percentage']:.1%})")
        report.append("")
        
        # Outliers section
        report.append("OUTLIER ANALYSIS:")
        report.append("-" * 30)
        for col, info in results['outliers'].items():
            status = "✓" if info['percentage'] <= 0.05 else "⚠"
            report.append(f"{status} {col}: {info['count']} outliers ({info['percentage']:.1%})")
        report.append("")
        
        # Recommendations section
        if results['recommendations']:
            report.append("RECOMMENDATIONS:")
            report.append("-" * 30)
            for i, rec in enumerate(results['recommendations'], 1):
                report.append(f"{i}. {rec}")
        
        return "\n".join(report)