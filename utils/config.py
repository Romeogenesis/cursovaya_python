"""
Configuration module for the Metrological Equipment Analysis System.
Contains default weights, equipment profiles, and system constants.
"""
from typing import Dict, List, Any

# Default weights for technical level assessment (sum to 1.0)
DEFAULT_WEIGHTS: Dict[str, float] = {
    "accuracy": 0.30,      # Measurement accuracy (higher is better)
    "price": 0.20,         # Equipment price (lower is better for normalization)
    "digital_display": 0.15,  # Digital display presence (1 or 0)
    "temperature_range": 0.20,  # Operating temperature range (wider is better)
    "weight": 0.15         # Weight (lighter is better for portability)
}

# Equipment type profiles with reference values for normalization
EQUIPMENT_PROFILES: Dict[str, Dict[str, Any]] = {
    "oscilloscope": {
        "name": "Осциллограф",
        "reference_accuracy": 0.001,  # Best-in-class accuracy (V)
        "reference_price": 50000,     # Reference price (RUB)
        "max_temperature_range": 100,  # Max operating temp range (°C)
        "max_weight": 10              # Max weight (kg)
    },
    "generator": {
        "name": "Генератор сигналов",
        "reference_accuracy": 0.0001,  # Best-in-class frequency accuracy
        "reference_price": 80000,
        "max_temperature_range": 80,
        "max_weight": 15
    },
    "spectrometer": {
        "name": "Спектрометр",
        "reference_accuracy": 0.00001,  # Best-in-class wavelength accuracy
        "reference_price": 150000,
        "max_temperature_range": 60,
        "max_weight": 25
    },
    "multimeter": {
        "name": "Мультиметр",
        "reference_accuracy": 0.0001,  # Best-in-class measurement accuracy
        "reference_price": 10000,
        "max_temperature_range": 70,
        "max_weight": 2
    },
    "calibrator": {
        "name": "Калибратор",
        "reference_accuracy": 0.00005,  # Best-in-class calibration accuracy
        "reference_price": 120000,
        "max_temperature_range": 50,
        "max_weight": 8
    }
}

# Column mapping for dataset
COLUMN_MAPPING: Dict[str, str] = {
    "avg_equipment_price": "price",
    "measurement_accuracy": "accuracy",
    "is_digital_display": "digital_display",
    "operating_temperature": "temperature_range",
    "weight": "weight",
    "monthly_equipment_sales": "sales"
}

# Optimization constraints defaults
OPTIMIZATION_DEFAULTS: Dict[str, float] = {
    "budget": 500000,       # Total budget (RUB)
    "min_demand": 10,       # Minimum units to purchase per type
    "warehouse_capacity": 100  # Maximum storage capacity
}

# Regression model options
REGRESSION_MODELS: List[str] = ["linear", "polynomial"]

# Visualization settings
PLOTLY_TEMPLATE: str = "plotly_white"
EXPORT_FORMATS: List[str] = ["csv", "html", "png"]
