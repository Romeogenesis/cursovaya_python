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
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RegressionStrategy':
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        pass


class LinearRegressionStrategy(RegressionStrategy):
    def __init__(self):
        self.model = LinearRegression()
        self._metrics: Dict[str, float] = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressionStrategy':
        self.model.fit(X, y)
        self._metrics = self.get_metrics(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def get_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        y_pred = self.predict(X)
        return {
            'r2': r2_score(y, y_pred),
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred))
            
        }
    
    def get_coefficients(self) -> np.ndarray:
        return self.model.coef_
    
    def get_intercept(self) -> float:
        return self.model.intercept_


class PolynomialRegressionStrategy(RegressionStrategy):
    def __init__(self, degree: int = 2):
        self.degree = degree
        self.poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        self.model = LinearRegression()
        self._metrics: Dict[str, float] = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PolynomialRegressionStrategy':
        X_poly = self.poly_features.fit_transform(X)
        self.model.fit(X_poly, y)
        self._metrics = self.get_metrics(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_poly = self.poly_features.transform(X)
        return self.model.predict(X_poly)
    
    def get_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        y_pred = self.predict(X)
        print(np.sqrt(mean_squared_error(y, y_pred)))
        return {
            'r2': r2_score(y, y_pred),
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred))
        }


class ForecastEngine:
    def __init__(self, strategy: Optional[RegressionStrategy] = None):
        self.strategy = strategy or LinearRegressionStrategy()
        self._feature_names: List[str] = []
        self._target_name: str = 'sales'
        self._last_error: Optional[str] = None
    
    def set_strategy(self, strategy: RegressionStrategy):
        self.strategy = strategy
    
    def analyze_correlations(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> pd.DataFrame:
        try:
            df = data.copy()
            
            if target_column is None:
                target_column = self._target_name
            
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found")
            
            numeric_df = df.select_dtypes(include=[np.number])
            
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
        try:
            X = data[feature_columns].values
            y = data[target_column].values
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            self.strategy.fit(X_train, y_train)
            
            test_metrics = self.strategy.get_metrics(X_test, y_test)
            train_metrics = self.strategy.get_metrics(X_train, y_train)
            
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
        try:
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
        try:
            if variable_feature in fixed_features:
                del fixed_features[variable_feature]
            
            variable_values = np.linspace(min_val, max_val, n_points)
            
            results = []
            
            for val in variable_values:
                features = fixed_features.copy()
                features[variable_feature] = val
                
                for col in self._feature_names:
                    if col not in features:
                        features[col] = data[col].median()
                
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
        try:
            X = data[feature_columns].values
            y = data[target_column].values
            
            if isinstance(self.strategy, LinearRegressionStrategy):
                model = LinearRegression()
            else:
                poly_features = PolynomialFeatures(
                    degree=self.strategy.degree,
                    include_bias=False
                )
                X = poly_features.fit_transform(X)
                model = LinearRegression()
            
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
        corr_matrix = self.analyze_correlations(data, target_column)
        
        if corr_matrix.empty or target_column not in corr_matrix.columns:
            return []
        
        target_corr = corr_matrix[target_column].drop(target_column)
        
        significant = target_corr[abs(target_corr) >= threshold].index.tolist()
        
        return significant
    
    @property
    def last_error(self) -> Optional[str]:
        return self._last_error
    
    @property
    def feature_names(self) -> List[str]:
        return self._feature_names
