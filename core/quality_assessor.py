"""
Technical level assessment module.
Implements normalization, weighted scoring, and Strategy pattern for weight selection.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from utils.config import DEFAULT_WEIGHTS, EQUIPMENT_PROFILES
from utils.validators import validate_weights


class NormalizationStrategy(ABC):
    """Abstract base class for normalization strategies."""
    
    @abstractmethod
    def normalize(self, data: pd.DataFrame, column: str) -> pd.Series:
        """
        Normalize a column to [0, 1] range.
        
        Args:
            data: DataFrame containing the column
            column: Column name to normalize
            
        Returns:
            Normalized Series
        """
        pass


class MinMaxNormalization(NormalizationStrategy):
    """Min-Max normalization strategy."""
    
    def normalize(self, data: pd.DataFrame, column: str) -> pd.Series:
        """Normalize using min-max scaling."""
        min_val = data[column].min()
        max_val = data[column].max()
        
        if max_val - min_val == 0:
            return pd.Series([0.5] * len(data), index=data.index)
        
        return (data[column] - min_val) / (max_val - min_val)


class ReferenceNormalization(NormalizationStrategy):
    """Reference-based normalization strategy."""
    
    def __init__(self, reference_values: Dict[str, float]):
        """
        Initialize with reference values.
        
        Args:
            reference_values: Dictionary of column names to reference values
        """
        self.reference_values = reference_values
    
    def normalize(self, data: pd.DataFrame, column: str) -> pd.Series:
        """Normalize using reference value."""
        if column not in self.reference_values:
            # Fallback to min-max if no reference
            return MinMaxNormalization().normalize(data, column)
        
        ref_val = self.reference_values[column]
        if ref_val == 0:
            return pd.Series([0.5] * len(data), index=data.index)
        
        # For metrics where lower is better (like accuracy error)
        if column in ['accuracy', 'price']:
            normalized = ref_val / data[column]
            return normalized.clip(0, 1)
        else:
            # For metrics where higher is better
            normalized = data[column] / ref_val
            return normalized.clip(0, 1)


class WeightStrategy(ABC):
    """Abstract base class for weight selection strategies."""
    
    @abstractmethod
    def get_weights(self, equipment_type: str) -> Dict[str, float]:
        """
        Get weights for technical level calculation.
        
        Args:
            equipment_type: Type of equipment
            
        Returns:
            Dictionary of weight name to value
        """
        pass


class DefaultWeightStrategy(WeightStrategy):
    """Default weight strategy using predefined weights."""
    
    def get_weights(self, equipment_type: str) -> Dict[str, float]:
        """Return default weights regardless of equipment type."""
        return DEFAULT_WEIGHTS.copy()


class CustomWeightStrategy(WeightStrategy):
    """Custom weight strategy with user-defined weights."""
    
    def __init__(self, weights: Dict[str, float]):
        """
        Initialize with custom weights.
        
        Args:
            weights: Custom weights dictionary
        """
        is_valid, error_msg = validate_weights(weights)
        if not is_valid:
            raise ValueError(error_msg)
        self.weights = weights.copy()
    
    def get_weights(self, equipment_type: str) -> Dict[str, float]:
        """Return custom weights."""
        return self.weights.copy()


class EquipmentSpecificWeightStrategy(WeightStrategy):
    """Equipment-specific weight strategy."""
    
    def __init__(self):
        """Initialize with equipment-specific weight profiles."""
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
        """Return equipment-specific weights or default."""
        return self.weight_profiles.get(equipment_type.lower(), DEFAULT_WEIGHTS.copy())


class QualityAssessor:
    """
    Technical level assessment class.
    Implements Strategy pattern for normalization and weighting.
    """
    
    def __init__(
        self,
        normalization_strategy: Optional[NormalizationStrategy] = None,
        weight_strategy: Optional[WeightStrategy] = None
    ):
        """
        Initialize quality assessor.
        
        Args:
            normalization_strategy: Strategy for normalizing values
            weight_strategy: Strategy for selecting weights
        """
        self.normalization_strategy = normalization_strategy or MinMaxNormalization()
        self.weight_strategy = weight_strategy or DefaultWeightStrategy()
        self._last_error: Optional[str] = None
    
    def set_normalization_strategy(self, strategy: NormalizationStrategy):
        """Set normalization strategy."""
        self.normalization_strategy = strategy
    
    def set_weight_strategy(self, strategy: WeightStrategy):
        """Set weight strategy."""
        self.weight_strategy = strategy
    
    def calculate_technical_level(
        self,
        data: pd.DataFrame,
        equipment_type: str = 'oscilloscope',
        higher_is_better_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """
        Calculate comprehensive technical level score.
        
        Args:
            data: DataFrame with equipment characteristics
            equipment_type: Type of equipment for weight selection
            higher_is_better_cols: Columns where higher values are better
            
        Returns:
            Tuple of (DataFrame with scores, error message)
        """
        try:
            df = data.copy()
            weights = self.weight_strategy.get_weights(equipment_type)
            
            # Define which columns benefit from higher values
            if higher_is_better_cols is None:
                higher_is_better_cols = ['accuracy', 'digital_display', 'temperature_range']
            
            # Columns where lower values are better
            lower_is_better_cols = ['price', 'weight']
            
            normalized_scores = {}
            
            # Normalize each metric
            for col in weights.keys():
                if col not in df.columns:
                    continue
                
                if col in higher_is_better_cols:
                    # Higher is better - direct normalization
                    normalized = self._normalize_column(df, col, invert=False)
                elif col in lower_is_better_cols:
                    # Lower is better - invert normalization
                    normalized = self._normalize_column(df, col, invert=True)
                else:
                    # Default: higher is better
                    normalized = self._normalize_column(df, col, invert=False)
                
                normalized_scores[col] = normalized
            
            # Calculate weighted sum
            technical_level = pd.Series([0.0] * len(df), index=df.index)
            
            for col, weight in weights.items():
                if col in normalized_scores:
                    technical_level += weight * normalized_scores[col]
            
            df['technical_level'] = technical_level
            
            # Add individual normalized scores for visualization
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
        """
        Normalize a single column.
        
        Args:
            data: DataFrame containing the column
            column: Column name
            invert: Whether to invert the result (for "lower is better" metrics)
            
        Returns:
            Normalized Series
        """
        normalized = self.normalization_strategy.normalize(data, column)
        
        if invert:
            return 1 - normalized
        
        return normalized
    
    def get_reference_profile(self, equipment_type: str) -> Dict[str, float]:
        """
        Get reference profile for equipment type.
        
        Args:
            equipment_type: Type of equipment
            
        Returns:
            Dictionary of reference values
        """
        return EQUIPMENT_PROFILES.get(equipment_type.lower(), {})
    
    def compare_with_reference(
        self,
        data: pd.DataFrame,
        equipment_type: str
    ) -> pd.DataFrame:
        """
        Compare equipment with reference profile.
        
        Args:
            data: DataFrame with equipment data
            equipment_type: Type of equipment
            
        Returns:
            DataFrame with comparison results
        """
        df = data.copy()
        reference = self.get_reference_profile(equipment_type)
        
        if not reference:
            return df
        
        # Calculate similarity to reference (1 = identical, 0 = completely different)
        similarity_scores = []
        
        for idx, row in df.iterrows():
            score = 0
            count = 0
            
            for metric, ref_val in reference.items():
                if metric in row:
                    actual_val = row[metric]
                    if ref_val != 0:
                        # Calculate similarity ratio
                        sim = min(actual_val / ref_val, ref_val / actual_val)
                        score += sim
                        count += 1
            
            avg_similarity = score / count if count > 0 else 0
            similarity_scores.append(avg_similarity)
        
        df['reference_similarity'] = similarity_scores
        return df
    
    @property
    def last_error(self) -> Optional[str]:
        """Get last error message."""
        return self._last_error
