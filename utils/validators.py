from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np


def validate_numeric_range(
    value: float, 
    min_val: float, 
    max_val: float, 
    field_name: str = "value"
) -> Tuple[bool, str]:
    try:
        val = float(value)
        if val < min_val or val > max_val:
            return False, f"{field_name} must be between {min_val} and {max_val}, got {val}"
        return True, ""
    except (TypeError, ValueError):
        return False, f"{field_name} must be a numeric value"


def validate_dataframe_structure(
    df: pd.DataFrame, 
    required_columns: List[str]
) -> Tuple[bool, str]:
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return False, f"Missing required columns: {', '.join(missing_cols)}"
    return True, ""


def validate_weights(weights: Dict[str, float]) -> Tuple[bool, str]:
    if not weights:
        return False, "Weights dictionary cannot be empty"
    
    total = sum(weights.values())
    if abs(total - 1.0) > 0.01:
        return False, f"Weights must sum to 1.0, current sum: {total:.4f}"
    
    for name, val in weights.items():
        if val < 0:
            return False, f"Weight '{name}' cannot be negative: {val}"
        if val > 1:
            return False, f"Weight '{name}' cannot exceed 1.0: {val}"
    
    return True, ""


def validate_optimization_constraints(
    budget: float,
    min_demand: int,
    warehouse_capacity: int
) -> Tuple[bool, str]:
    if budget <= 0:
        return False, "Budget must be positive"
    if min_demand < 0:
        return False, "Minimum demand cannot be negative"
    if warehouse_capacity <= 0:
        return False, "Warehouse capacity must be positive"
    if min_demand * 5 > warehouse_capacity:
        return False, "Minimum demand too high for warehouse capacity"
    
    return True, ""


def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    column_types = {}
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            column_types[col] = 'numeric'
        elif df[col].dtype == 'bool':
            column_types[col] = 'boolean'
        elif df[col].nunique() <= 10:
            column_types[col] = 'categorical'
        else:
            column_types[col] = 'text'
    return column_types


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    
    df_clean = df_clean.dropna(how='all')
    
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    for col in numeric_cols:
        mean = df_clean[col].mean()
        std = df_clean[col].std()
        if std > 0:
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
            df_clean = df_clean[(df_clean[col] >= lower_bound) & 
                               (df_clean[col] <= upper_bound)]
    
    return df_clean.reset_index(drop=True)


def validate_equipment_type(equipment_type: str, valid_types: List[str]) -> Tuple[bool, str]:
    if not equipment_type:
        return False, "Equipment type cannot be empty"
    if equipment_type.lower() not in [t.lower() for t in valid_types]:
        return False, f"Invalid equipment type. Choose from: {', '.join(valid_types)}"
    return True, ""