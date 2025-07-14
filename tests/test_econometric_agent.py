"""
Basic tests for the Econometric Agent.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from econometric_agent import EconometricAgent, EconomicDataFetcher, DataValidator, EconometricModels


class TestEconometricAgent(unittest.TestCase):
    """Test the main EconometricAgent class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = EconometricAgent()
        
        # Create synthetic test data
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='M')
        np.random.seed(42)
        
        self.test_data = pd.DataFrame({
            'GDP': 100 + np.cumsum(np.random.normal(0.3, 2, len(dates))),
            'UNEMPLOYMENT': np.clip(5 + np.random.normal(0, 1.5, len(dates)), 2, 15),
            'INFLATION': 2 + np.random.normal(0, 0.8, len(dates)),
            'INTEREST_RATE': np.clip(2 + np.random.normal(0, 0.5, len(dates)), 0, 8)
        }, index=dates)
        
        # Store test data in agent
        self.agent.datasets['test'] = self.test_data
        self.agent.current_data = self.test_data
    
    def test_data_validation(self):
        """Test data validation functionality."""
        results = self.agent.validate_data('test')
        
        self.assertIn('quality_score', results)
        self.assertIn('missing_data', results)
        self.assertIn('outliers', results)
        self.assertIsInstance(results['quality_score'], float)
        self.assertGreaterEqual(results['quality_score'], 0)
        self.assertLessEqual(results['quality_score'], 100)
    
    def test_data_cleaning(self):
        """Test data cleaning functionality."""
        cleaned_data = self.agent.clean_data('test')
        
        self.assertIsInstance(cleaned_data, pd.DataFrame)
        self.assertEqual(cleaned_data.shape[1], self.test_data.shape[1])
        self.assertIn('test_cleaned', self.agent.datasets)
    
    def test_linear_regression(self):
        """Test linear regression model building."""
        results = self.agent.build_regression_model(
            target_variable='GDP',
            feature_variables=['UNEMPLOYMENT', 'INFLATION'],
            dataset_name='test'
        )
        
        self.assertIn('r2_test', results)
        self.assertIn('mse_test', results)
        self.assertIn('model_name', results)
        self.assertIsInstance(results['r2_test'], float)
    
    def test_stationarity_analysis(self):
        """Test stationarity analysis."""
        results = self.agent.analyze_stationarity('GDP', 'test')
        
        self.assertIn('adf_test', results)
        self.assertIn('kpss_test', results)
        self.assertIn('interpretation', results)


class TestDataValidator(unittest.TestCase):
    """Test the DataValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = DataValidator()
        
        # Create test data with known issues
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'normal': np.random.normal(0, 1, 100),
            'with_missing': np.random.normal(0, 1, 100),
            'with_outliers': np.random.normal(0, 1, 100)
        }, index=dates)
        
        # Introduce missing values (not at the start for better interpolation)
        data.loc[data.index[20:30], 'with_missing'] = np.nan
        
        # Introduce outliers
        data.loc[data.index[95:], 'with_outliers'] = 10
        
        self.test_data = data
    
    def test_validation(self):
        """Test data validation."""
        results = self.validator.validate_data_quality(self.test_data)
        
        self.assertIn('missing_data', results)
        self.assertIn('outliers', results)
        self.assertIn('quality_score', results)
        
        # Check that missing data is detected
        self.assertGreater(results['missing_data']['with_missing']['count'], 0)
        
        # Check that outliers are detected
        self.assertGreater(results['outliers']['with_outliers']['count'], 0)
    
    def test_cleaning(self):
        """Test data cleaning."""
        cleaned_data = self.validator.clean_data(
            self.test_data,
            interpolate_missing=True,
            remove_outliers=True
        )
        
        self.assertIsInstance(cleaned_data, pd.DataFrame)
        self.assertEqual(cleaned_data.shape, self.test_data.shape)
        
        # Check that missing values are handled
        self.assertEqual(cleaned_data['with_missing'].isnull().sum(), 0)


class TestEconometricModels(unittest.TestCase):
    """Test the EconometricModels class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.models = EconometricModels()
        
        # Create synthetic test data
        np.random.seed(42)
        n_obs = 100
        
        self.test_data = pd.DataFrame({
            'y': np.random.normal(0, 1, n_obs),
            'x1': np.random.normal(0, 1, n_obs),
            'x2': np.random.normal(0, 1, n_obs)
        })
        
        # Make y dependent on x1 and x2
        self.test_data['y'] = 2 + 0.5 * self.test_data['x1'] + 0.3 * self.test_data['x2'] + np.random.normal(0, 0.5, n_obs)
    
    def test_linear_regression(self):
        """Test linear regression."""
        results = self.models.linear_regression(
            self.test_data, 'y', ['x1', 'x2']
        )
        
        self.assertIn('r2_test', results)
        self.assertIn('coefficients', results)
        self.assertIn('p_values', results)
        self.assertIsInstance(results['r2_test'], float)
    
    def test_regularized_regression(self):
        """Test regularized regression."""
        results = self.models.regularized_regression(
            self.test_data, 'y', ['x1', 'x2'], method='ridge'
        )
        
        self.assertIn('r2_test', results)
        self.assertIn('coefficients', results)
        self.assertEqual(results['method'], 'ridge')
    
    def test_model_comparison(self):
        """Test model comparison."""
        # Build multiple models
        self.models.linear_regression(self.test_data, 'y', ['x1', 'x2'])
        self.models.regularized_regression(self.test_data, 'y', ['x1', 'x2'], method='ridge')
        
        comparison = self.models.model_comparison()
        
        self.assertIsInstance(comparison, pd.DataFrame)
        self.assertGreater(len(comparison), 0)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEconometricAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestDataValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestEconometricModels))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    print(f"\nTests {'PASSED' if success else 'FAILED'}")