from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from utils.config import DEFAULT_WEIGHTS, EQUIPMENT_PROFILES
from utils.validators import validate_weights


class NormalizationStrategy(ABC):
    @abstractmethod
    def normalize(self, data: pd.DataFrame, column: str) -> pd.Series:
        pass


class MinMaxNormalization(NormalizationStrategy):
    def normalize(self, data: pd.DataFrame, column: str) -> pd.Series:
        min_val = data[column].min()
        max_val = data[column].max()
        
        if max_val - min_val == 0:
            return pd.Series([0.5] * len(data), index=data.index)
        
        return (data[column] - min_val) / (max_val - min_val)


class ReferenceNormalization(NormalizationStrategy):
    def __init__(self, reference_values: Dict[str, float]):
        self.reference_values = reference_values
    
    def normalize(self, data: pd.DataFrame, column: str) -> pd.Series:
        if column not in self.reference_values:
            return MinMaxNormalization().normalize(data, column)
        
        ref_val = self.reference_values[column]
        if ref_val == 0:
            return pd.Series([0.5] * len(data), index=data.index)
        
        if column in ['accuracy', 'price']:
            normalized = ref_val / data[column]
            return normalized.clip(0, 1)
        else:
            normalized = data[column] / ref_val
            return normalized.clip(0, 1)


class WeightStrategy(ABC):
    @abstractmethod
    def get_weights(self, equipment_type: str) -> Dict[str, float]:
        pass


class DefaultWeightStrategy(WeightStrategy):
    def get_weights(self, equipment_type: str) -> Dict[str, float]:
        return DEFAULT_WEIGHTS.copy()


class CustomWeightStrategy(WeightStrategy):
    def __init__(self, weights: Dict[str, float]):
        is_valid, error_msg = validate_weights(weights)
        if not is_valid:
            raise ValueError(error_msg)
        self.weights = weights.copy()
    
    def get_weights(self, equipment_type: str) -> Dict[str, float]:
        return self.weights.copy()


class EquipmentSpecificWeightStrategy(WeightStrategy):
    def __init__(self):
        self.weight_profiles = {
            'oscilloscope': {
                'accuracy': 0.35,
                'price': 0.20,
                'digital_display': 0.15,
                'temperature_range': 0.15,
                'weight': 0.15
            },
            'generator': {
                'accuracy': 0.30,
                'price': 0.25,
                'digital_display': 0.10,
                'temperature_range': 0.20,
                'weight': 0.15
            },
            'spectrometer': {
                'accuracy': 0.40,
                'price': 0.15,
                'digital_display': 0.10,
                'temperature_range': 0.20,
                'weight': 0.15
            }
        }
    
    def get_weights(self, equipment_type: str) -> Dict[str, float]:
        return self.weight_profiles.get(equipment_type.lower(), DEFAULT_WEIGHTS.copy())


class QualityAssessor:
    def __init__(
        self,
        normalization_strategy: Optional[NormalizationStrategy] = None,
        weight_strategy: Optional[WeightStrategy] = None
    ):
        self.normalization_strategy = normalization_strategy or MinMaxNormalization()
        self.weight_strategy = weight_strategy or DefaultWeightStrategy()
        self._last_error: Optional[str] = None
    
    def set_normalization_strategy(self, strategy: NormalizationStrategy):
        self.normalization_strategy = strategy
    
    def set_weight_strategy(self, strategy: WeightStrategy):
        self.weight_strategy = strategy
    
    def calculate_technical_level(
        self,
        data: pd.DataFrame,
        equipment_type: str = 'oscilloscope',
        higher_is_better_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        try:
            df = data.copy()
            weights = self.weight_strategy.get_weights(equipment_type)
            
            if higher_is_better_cols is None:
                higher_is_better_cols = ['accuracy', 'digital_display', 'temperature_range']
            
            lower_is_better_cols = ['price', 'weight']
            
            normalized_scores = {}
            
            for col in weights.keys():
                if col not in df.columns:
                    continue
                
                if col in higher_is_better_cols:
                    normalized = self._normalize_column(df, col, invert=False)
                elif col in lower_is_better_cols:
                    normalized = self._normalize_column(df, col, invert=True)
                else:
                    normalized = self._normalize_column(df, col, invert=False)
                
                normalized_scores[col] = normalized
            
            technical_level = pd.Series([0.0] * len(df), index=df.index)
            
            for col, weight in weights.items():
                if col in normalized_scores:
                    technical_level += weight * normalized_scores[col]
            
            df['technical_level'] = technical_level
            
            for col, normalized in normalized_scores.items():
                df[f'norm_{col}'] = normalized
            
            return df, None
            
        except Exception as e:
            self._last_error = f"Error calculating technical level: {str(e)}"
            return data, self._last_error
    
    def _normalize_column(
        self,
        data: pd.DataFrame,
        column: str,
        invert: bool = False
    ) -> pd.Series:
        normalized = self.normalization_strategy.normalize(data, column)
        
        if invert:
            return 1 - normalized
        
        return normalized
    
    def get_reference_profile(self, equipment_type: str) -> Dict[str, float]:
        return EQUIPMENT_PROFILES.get(equipment_type.lower(), {})
    
    def compare_with_reference(
        self,
        data: pd.DataFrame,
        equipment_type: str
    ) -> pd.DataFrame:
        df = data.copy()
        reference = self.get_reference_profile(equipment_type)
        
        if not reference:
            return df
        
        similarity_scores = []
        
        for idx, row in df.iterrows():
            score = 0
            count = 0
            
            for metric, ref_val in reference.items():
                if metric in row:
                    actual_val = row[metric]
                    if ref_val != 0:
                        sim = min(actual_val / ref_val, ref_val / actual_val)
                        score += sim
                        count += 1
            
            avg_similarity = score / count if count > 0 else 0
            similarity_scores.append(avg_similarity)
        
        df['reference_similarity'] = similarity_scores
        return df
    
    @property
    def last_error(self) -> Optional[str]:
        return self._last_error