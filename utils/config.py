from typing import Dict, List, Any

DEFAULT_WEIGHTS: Dict[str, float] = {
    "accuracy": 0.30,
    "price": 0.20,
    "digital_display": 0.15,
    "temperature_range": 0.20,
    "weight": 0.15
}

EQUIPMENT_PROFILES: Dict[str, Dict[str, Any]] = {
    "oscilloscope": {
        "name": "Осциллограф",
        "reference_accuracy": 0.001,
        "reference_price": 50000,
        "max_temperature_range": 100,
        "max_weight": 10
    },
    "generator": {
        "name": "Генератор сигналов",
        "reference_accuracy": 0.0001,
        "reference_price": 80000,
        "max_temperature_range": 80,
        "max_weight": 15
    },
    "spectrometer": {
        "name": "Спектрометр",
        "reference_accuracy": 0.00001,
        "reference_price": 150000,
        "max_temperature_range": 60,
        "max_weight": 25
    },
    "multimeter": {
        "name": "Мультиметр",
        "reference_accuracy": 0.0001,
        "reference_price": 10000,
        "max_temperature_range": 70,
        "max_weight": 2
    },
    "calibrator": {
        "name": "Калибратор",
        "reference_accuracy": 0.00005,
        "reference_price": 120000,
        "max_temperature_range": 50,
        "max_weight": 8
    }
}

COLUMN_MAPPING: Dict[str, str] = {
    "avg_equipment_price": "price",
    "measurement_accuracy": "accuracy",
    "is_digital_display": "digital_display",
    "operating_temperature": "temperature_range",
    "weight": "weight",
    "monthly_equipment_sales": "sales"
}

OPTIMIZATION_DEFAULTS: Dict[str, float] = {
    "budget": 500000,
    "min_demand": 10,
    "warehouse_capacity": 100
}

REGRESSION_MODELS: List[str] = ["linear", "polynomial"]

PLOTLY_TEMPLATE: str = "plotly_white"
EXPORT_FORMATS: List[str] = ["csv", "html", "png"]