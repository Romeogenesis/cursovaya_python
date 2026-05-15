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
        return self._data
    
    @property
    def last_error(self) -> Optional[str]:
        return self._last_error
    
    def load_csv(self, filepath: str) -> bool:
        try:
            if not os.path.exists(filepath):
                self._last_error = f"File not found: {filepath}"
                return False
            
            df = pd.read_csv(filepath)

            return self._process_dataframe(df)

        except PermissionError as e:
            self._last_error = f"Permission error loading CSV: {str(e)}. Файл может быть занят другим процессом."
            return False
            
        except Exception as e:
            self._last_error = f"Error loading CSV: {str(e)}"
            return False
    
    def load_json(self, filepath: str) -> bool:
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
        try:
            df_clean = clean_dataframe(df)
            
            if df_clean.empty:
                self._last_error = "DataFrame is empty after cleaning"
                return False
            
            df_mapped = self._map_columns(df_clean)
            
            required_cols = ['price', 'accuracy', 'digital_display', 
                           'temperature_range', 'weight', 'sales']
            is_valid, error_msg = validate_dataframe_structure(df_mapped, required_cols)
            
            if not is_valid:
                self._last_error = error_msg
                return False
            
            col_types = detect_column_types(df_mapped)
            
            self._data = df_mapped
            self._last_error = None
            return True
            
        except Exception as e:
            self._last_error = f"Error processing DataFrame: {str(e)}"
            return False
    
    def _map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df_mapped = df.copy()
        
        for orig_col, std_col in self._column_mapping.items():
            if orig_col in df_mapped.columns and std_col not in df_mapped.columns:
                df_mapped[std_col] = df_mapped[orig_col]
        
        return df_mapped
    
    def get_numeric_columns(self) -> List[str]:
        if self._data is None:
            return []
        return self._data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    def get_categorical_columns(self) -> List[str]:
        if self._data is None:
            return []
        return self._data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    def get_data_summary(self) -> Dict[str, any]:
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
        self._data = None
        self._last_error = None


def create_sample_dataset(filepath: str, n_samples: int = 100) -> bool:
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
