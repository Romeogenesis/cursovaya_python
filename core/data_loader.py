"""
Data loading and validation module.
Handles CSV/JSON file loading, column mapping, and data cleaning.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import json
import os

from utils.config import COLUMN_MAPPING
from utils.validators import (
    validate_dataframe_structure,
    detect_column_types,
    clean_dataframe
)


class DataLoader:
    """
    Data loader class for loading and validating metrological equipment data.
    Implements singleton pattern for consistent data access.
    """
    
    _instance = None
    _data: Optional[pd.DataFrame] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self._column_mapping = COLUMN_MAPPING.copy()
        self._last_error: Optional[str] = None
    
    @property
    def data(self) -> Optional[pd.DataFrame]:
        """Get the loaded data."""
        return self._data
    
    @property
    def last_error(self) -> Optional[str]:
        """Get the last error message."""
        return self._last_error
    
    def load_csv(self, filepath: str) -> bool:
        """
        Load data from CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(filepath):
                self._last_error = f"File not found: {filepath}"
                return False
            
            df = pd.read_csv(filepath)
            return self._process_dataframe(df)
            
        except Exception as e:
            self._last_error = f"Error loading CSV: {str(e)}"
            return False
    
    def load_json(self, filepath: str) -> bool:
        """
        Load data from JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(filepath):
                self._last_error = f"File not found: {filepath}"
                return False
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                self._last_error = "Invalid JSON structure"
                return False
            
            return self._process_dataframe(df)
            
        except Exception as e:
            self._last_error = f"Error loading JSON: {str(e)}"
            return False
    
    def load_from_dict(self, data_dict: Dict[str, Union[List, Any]]) -> bool:
        """
        Load data from dictionary.
        
        Args:
            data_dict: Dictionary with column names as keys and lists/values as values
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if all(isinstance(v, list) for v in data_dict.values()):
                df = pd.DataFrame(data_dict)
            else:
                df = pd.DataFrame([data_dict])
            
            return self._process_dataframe(df)
            
        except Exception as e:
            self._last_error = f"Error loading from dict: {str(e)}"
            return False
    
    def _process_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Process and validate DataFrame.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Clean the data
            df_clean = clean_dataframe(df)
            
            if df_clean.empty:
                self._last_error = "DataFrame is empty after cleaning"
                return False
            
            # Map columns if needed
            df_mapped = self._map_columns(df_clean)
            
            # Validate structure
            required_cols = ['price', 'accuracy', 'digital_display', 
                           'temperature_range', 'weight', 'sales']
            is_valid, error_msg = validate_dataframe_structure(df_mapped, required_cols)
            
            if not is_valid:
                self._last_error = error_msg
                return False
            
            # Detect column types
            col_types = detect_column_types(df_mapped)
            
            self._data = df_mapped
            self._last_error = None
            return True
            
        except Exception as e:
            self._last_error = f"Error processing DataFrame: {str(e)}"
            return False
    
    def _map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map original column names to standard names.
        
        Args:
            df: DataFrame with original column names
            
        Returns:
            DataFrame with mapped column names
        """
        df_mapped = df.copy()
        
        for orig_col, std_col in self._column_mapping.items():
            if orig_col in df_mapped.columns and std_col not in df_mapped.columns:
                df_mapped[std_col] = df_mapped[orig_col]
        
        return df_mapped
    
    def get_numeric_columns(self) -> List[str]:
        """Get list of numeric columns."""
        if self._data is None:
            return []
        return self._data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    def get_categorical_columns(self) -> List[str]:
        """Get list of categorical columns."""
        if self._data is None:
            return []
        return self._data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    def get_data_summary(self) -> Dict[str, any]:
        """Get summary statistics of loaded data."""
        if self._data is None:
            return {}
        
        return {
            'rows': len(self._data),
            'columns': len(self._data.columns),
            'column_names': list(self._data.columns),
            'numeric_columns': self.get_numeric_columns(),
            'categorical_columns': self.get_categorical_columns(),
            'missing_values': self._data.isnull().sum().to_dict()
        }
    
    def clear_data(self):
        """Clear loaded data."""
        self._data = None
        self._last_error = None


def create_sample_dataset(filepath: str, n_samples: int = 100) -> bool:
    """
    Create a sample dataset for testing.
    
    Args:
        filepath: Path to save the dataset
        n_samples: Number of samples to generate
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import numpy as np
        
        np.random.seed(42)
        
        data = {
            'avg_equipment_price': np.random.uniform(5000, 200000, n_samples),
            'measurement_accuracy': np.random.uniform(0.0001, 0.01, n_samples),
            'is_digital_display': np.random.choice([0, 1], n_samples),
            'operating_temperature': np.random.uniform(-20, 80, n_samples),
            'weight': np.random.uniform(0.5, 30, n_samples),
            'monthly_equipment_sales': np.random.randint(10, 500, n_samples)
        }
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        return True
        
    except Exception as e:
        print(f"Error creating sample dataset: {e}")
        return False
