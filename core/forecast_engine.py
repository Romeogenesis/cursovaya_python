"""
Forecast engine module.
Implements correlation analysis, regression models, and interval analysis.
Uses Strategy pattern for model selection.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from scipy import stats

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error


class RegressionStrategy(ABC):
    """Abstract base class for regression strategies."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RegressionStrategy':
        """Fit the model to training data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def get_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Get model performance metrics."""
        pass


class LinearRegressionStrategy(RegressionStrategy):
    """Linear regression strategy."""
    
    def __init__(self):
        self.model = LinearRegression()
        self._metrics: Dict[str, float] = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressionStrategy':
        """Fit linear regression model."""
        self.model.fit(X, y)
        self._metrics = self.get_metrics(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using linear model."""
        return self.model.predict(X)
    
    def get_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Calculate R² and MSE."""
        y_pred = self.predict(X)
        return {
            'r2': r2_score(y, y_pred),
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred))
        }
    
    def get_coefficients(self) -> np.ndarray:
        """Get model coefficients."""
        return self.model.coef_
    
    def get_intercept(self) -> float:
        """Get model intercept."""
        return self.model.intercept_


class PolynomialRegressionStrategy(RegressionStrategy):
    """Polynomial regression strategy."""
    
    def __init__(self, degree: int = 2):
        self.degree = degree
        self.poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        self.model = LinearRegression()
        self._metrics: Dict[str, float] = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PolynomialRegressionStrategy':
        """Fit polynomial regression model."""
        X_poly = self.poly_features.fit_transform(X)
        self.model.fit(X_poly, y)
        self._metrics = self.get_metrics(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using polynomial model."""
        X_poly = self.poly_features.transform(X)
        return self.model.predict(X_poly)
    
    def get_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Calculate R² and MSE."""
        y_pred = self.predict(X)
        return {
            'r2': r2_score(y, y_pred),
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred))
        }


class ForecastEngine:
    """
    Forecast engine for predicting equipment dynamics.
    Implements Strategy pattern for regression model selection.
    """
    
    def __init__(self, strategy: Optional[RegressionStrategy] = None):
        """
        Initialize forecast engine.
        
        Args:
            strategy: Regression strategy to use
        """
        self.strategy = strategy or LinearRegressionStrategy()
        self._feature_names: List[str] = []
        self._target_name: str = 'sales'
        self._last_error: Optional[str] = None
    
    def set_strategy(self, strategy: RegressionStrategy):
        """Set regression strategy."""
        self.strategy = strategy
    
    def analyze_correlations(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Analyze correlations between features and target variable.
        
        Args:
            data: DataFrame with equipment data
            target_column: Target column name
            
        Returns:
            DataFrame with correlation matrix
        """
        try:
            df = data.copy()
            
            if target_column is None:
                target_column = self._target_name
            
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found")
            
            # Select only numeric columns
            numeric_df = df.select_dtypes(include=[np.number])
            
            # Calculate correlation matrix
            corr_matrix = numeric_df.corr()
            
            return corr_matrix
            
        except Exception as e:
            self._last_error = f"Error analyzing correlations: {str(e)}"
            return pd.DataFrame()
    
    def fit(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[Dict[str, float], Optional[str]]:
        """
        Fit regression model to data.
        
        Args:
            data: DataFrame with equipment data
            feature_columns: List of feature column names
            target_column: Target column name
            test_size: Fraction of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (metrics dictionary, error message)
        """
        try:
            # Prepare data
            X = data[feature_columns].values
            y = data[target_column].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Fit model
            self.strategy.fit(X_train, y_train)
            
            # Evaluate on test set
            test_metrics = self.strategy.get_metrics(X_test, y_test)
            train_metrics = self.strategy.get_metrics(X_train, y_train)
            
            # Check for overfitting
            overfitting_gap = train_metrics['r2'] - test_metrics['r2']
            
            metrics = {
                'train_r2': train_metrics['r2'],
                'test_r2': test_metrics['r2'],
                'train_mse': train_metrics['mse'],
                'test_mse': test_metrics['mse'],
                'overfitting_gap': overfitting_gap,
                'is_overfitting': overfitting_gap > 0.1
            }
            
            self._feature_names = feature_columns
            self._target_name = target_column
            
            return metrics, None
            
        except Exception as e:
            self._last_error = f"Error fitting model: {str(e)}"
            return {}, self._last_error
    
    def predict(self, features: Dict[str, float]) -> Tuple[float, Optional[str]]:
        """
        Make prediction for given feature values.
        
        Args:
            features: Dictionary of feature name to value
            
        Returns:
            Tuple of (prediction, error message)
        """
        try:
            # Create feature vector in correct order
            X = np.array([[features[col] for col in self._feature_names]])
            
            prediction = self.strategy.predict(X)[0]
            
            return prediction, None
            
        except Exception as e:
            self._last_error = f"Error making prediction: {str(e)}"
            return 0.0, self._last_error
    
    def interval_analysis(
        self,
        data: pd.DataFrame,
        fixed_features: Dict[str, float],
        variable_feature: str,
        min_val: float,
        max_val: float,
        n_points: int = 50
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """
        Analyze how target variable changes with one feature while others are fixed.
        
        Args:
            data: DataFrame with equipment data
            fixed_features: Dictionary of fixed feature values
            variable_feature: Name of feature to vary
            min_val: Minimum value for variable feature
            max_val: Maximum value for variable feature
            n_points: Number of points to evaluate
            
        Returns:
            Tuple of (DataFrame with results, error message)
        """
        try:
            if variable_feature in fixed_features:
                del fixed_features[variable_feature]
            
            # Generate range of values for variable feature
            variable_values = np.linspace(min_val, max_val, n_points)
            
            results = []
            
            for val in variable_values:
                # Create feature vector
                features = fixed_features.copy()
                features[variable_feature] = val
                
                # Ensure all required features are present
                for col in self._feature_names:
                    if col not in features:
                        # Use median from data if not specified
                        features[col] = data[col].median()
                
                # Make prediction
                prediction, error = self.predict(features)
                
                if error:
                    continue
                
                results.append({
                    variable_feature: val,
                    'predicted_' + self._target_name: prediction
                })
            
            result_df = pd.DataFrame(results)
            
            return result_df, None
            
        except Exception as e:
            self._last_error = f"Error in interval analysis: {str(e)}"
            return pd.DataFrame(), self._last_error
    
    def cross_validate(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        target_column: str,
        cv_folds: int = 5
    ) -> Dict[str, float]:
        """
        Perform cross-validation on the model.
        
        Args:
            data: DataFrame with equipment data
            feature_columns: List of feature column names
            target_column: Target column name
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with cross-validation scores
        """
        try:
            X = data[feature_columns].values
            y = data[target_column].values
            
            # Create a fresh model for cross-validation
            if isinstance(self.strategy, LinearRegressionStrategy):
                model = LinearRegression()
            else:
                poly_features = PolynomialFeatures(
                    degree=self.strategy.degree,
                    include_bias=False
                )
                X = poly_features.fit_transform(X)
                model = LinearRegression()
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
            
            return {
                'cv_mean_r2': cv_scores.mean(),
                'cv_std_r2': cv_scores.std(),
                'cv_min_r2': cv_scores.min(),
                'cv_max_r2': cv_scores.max(),
                'cv_scores': cv_scores.tolist()
            }
            
        except Exception as e:
            self._last_error = f"Error in cross-validation: {str(e)}"
            return {}
    
    def get_significant_features(
        self,
        data: pd.DataFrame,
        target_column: str,
        threshold: float = 0.3
    ) -> List[str]:
        """
        Identify features significantly correlated with target.
        
        Args:
            data: DataFrame with equipment data
            target_column: Target column name
            threshold: Correlation threshold for significance
            
        Returns:
            List of significant feature names
        """
        corr_matrix = self.analyze_correlations(data, target_column)
        
        if corr_matrix.empty or target_column not in corr_matrix.columns:
            return []
        
        target_corr = corr_matrix[target_column].drop(target_column)
        
        significant = target_corr[abs(target_corr) >= threshold].index.tolist()
        
        return significant
    
    @property
    def last_error(self) -> Optional[str]:
        """Get last error message."""
        return self._last_error
    
    @property
    def feature_names(self) -> List[str]:
        """Get feature names used in model."""
        return self._feature_names
