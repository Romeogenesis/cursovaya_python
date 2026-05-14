"""
Resource optimization module.
Implements linear programming for minimizing procurement costs.
"""
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy.optimize import linprog

from utils.config import OPTIMIZATION_DEFAULTS
from utils.validators import validate_optimization_constraints


class ResourceOptimizer:
    """
    Resource optimizer using linear programming.
    Minimizes procurement costs subject to constraints.
    """
    
    def __init__(self):
        """Initialize resource optimizer."""
        self._last_error: Optional[str] = None
        self._optimization_result: Optional[Dict] = None
    
    def optimize_procurement(
        self,
        equipment_data: pd.DataFrame,
        budget: float = OPTIMIZATION_DEFAULTS['budget'],
        min_demand: int = int(OPTIMIZATION_DEFAULTS['min_demand']),
        warehouse_capacity: int = int(OPTIMIZATION_DEFAULTS['warehouse_capacity'])
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """
        Optimize equipment procurement to minimize costs.
        
        Objective: Min Z = Σ(c_i * x_i)
        
        Constraints:
        - Total cost <= budget
        - Each equipment type >= min_demand
        - Total units <= warehouse_capacity
        
        Args:
            equipment_data: DataFrame with equipment characteristics
            budget: Total budget available
            min_demand: Minimum units per equipment type
            warehouse_capacity: Maximum storage capacity
            
        Returns:
            Tuple of (DataFrame with optimal quantities, error message)
        """
        try:
            # Validate constraints
            is_valid, error_msg = validate_optimization_constraints(
                budget, min_demand, warehouse_capacity
            )
            if not is_valid:
                return pd.DataFrame(), error_msg
            
            # Group data by unique equipment types or use row index
            # For dataset without explicit types, we aggregate into representative types
            n_types = min(5, len(equipment_data))  # Limit to 5 types for feasibility
            
            # Calculate average cost per "type" (using price column)
            if 'price' in equipment_data.columns:
                costs = equipment_data['price'].values[:n_types]
            elif 'avg_equipment_price' in equipment_data.columns:
                costs = equipment_data['avg_equipment_price'].values[:n_types]
            else:
                numeric_cols = equipment_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    costs = equipment_data[numeric_cols[0]].values[:n_types]
                else:
                    costs = np.ones(n_types) * 10000
            
            equipment_types = [f"Type_{i+1}" for i in range(n_types)]
            
            # Ensure costs are positive and reasonable
            costs = np.maximum(costs, 1000)  # Minimum cost of 1000
            
            # Check if problem is feasible
            min_total_cost = sum(costs) * min_demand
            if min_total_cost > budget:
                # Adjust min_demand to make problem feasible
                min_demand = max(1, int(budget / (sum(costs) + 1)))
            
            if n_types * min_demand > warehouse_capacity:
                # Adjust warehouse capacity or min_demand
                min_demand = max(1, warehouse_capacity // n_types)
            
            # Objective function coefficients (costs to minimize)
            c = costs
            
            # Inequality constraints (A_ub * x <= b_ub)
            A_ub = []
            b_ub = []
            
            # Budget constraint: sum(cost_i * x_i) <= budget
            A_ub.append(c)
            b_ub.append(budget)
            
            # Warehouse capacity: sum(x_i) <= warehouse_capacity
            A_ub.append(np.ones(n_types))
            b_ub.append(warehouse_capacity)
            
            # Bounds for each variable (min_demand <= x_i <= infinity)
            bounds = [(min_demand, None) for _ in range(n_types)]
            
            # Solve linear programming problem
            result = linprog(
                c=c,
                A_ub=np.array(A_ub),
                b_ub=np.array(b_ub),
                A_eq=None,
                b_eq=None,
                bounds=bounds,
                method='highs'
            )
            
            if not result.success:
                self._last_error = f"Optimization failed: {result.message}"
                # Return a fallback solution
                fallback_qty = [min_demand] * n_types
                results_df = pd.DataFrame({
                    'equipment_type': equipment_types,
                    'unit_cost': costs,
                    'optimal_quantity': fallback_qty,
                    'total_cost': costs * np.array(fallback_qty)
                })
                return results_df, self._last_error
            
            # Create results DataFrame
            optimal_quantities = np.round(result.x).astype(int)
            
            results_df = pd.DataFrame({
                'equipment_type': equipment_types,
                'unit_cost': costs,
                'optimal_quantity': optimal_quantities,
                'total_cost': costs * optimal_quantities
            })
            
            # Add comparison with naive equal distribution
            naive_quantity = max(min_demand, warehouse_capacity // n_types)
            results_df['naive_quantity'] = naive_quantity
            results_df['naive_total_cost'] = costs * naive_quantity
            
            # Store optimization result
            self._optimization_result = {
                'total_optimized_cost': results_df['total_cost'].sum(),
                'total_naive_cost': results_df['naive_total_cost'].sum(),
                'savings': results_df['naive_total_cost'].sum() - results_df['total_cost'].sum(),
                'budget_used': results_df['total_cost'].sum() / budget * 100,
                'warehouse_used': optimal_quantities.sum() / warehouse_capacity * 100
            }
            
            return results_df, None
            
        except Exception as e:
            self._last_error = f"Error in optimization: {str(e)}"
            return pd.DataFrame(), self._last_error
    
    def optimize_with_priority(
        self,
        equipment_data: pd.DataFrame,
        priorities: Dict[str, float],
        budget: float = OPTIMIZATION_DEFAULTS['budget']
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """
        Optimize procurement with priority weights.
        
        Args:
            equipment_data: DataFrame with equipment data
            priorities: Dictionary of equipment type to priority weight
            budget: Total budget
            
        Returns:
            Tuple of (results DataFrame, error message)
        """
        try:
            # Modify costs by priority (higher priority = lower effective cost)
            df = equipment_data.copy()
            
            if 'equipment_type' not in df.columns:
                df['equipment_type'] = [f"Type_{i}" for i in range(len(df))]
            
            # Adjust costs based on priority
            adjusted_costs = []
            for idx, row in df.iterrows():
                eq_type = row['equipment_type']
                base_cost = row.get('price', 10000)
                priority = priorities.get(eq_type, 1.0)
                
                # Higher priority means we want more of it, so reduce effective cost
                adjusted_cost = base_cost / (priority + 0.1)
                adjusted_costs.append(adjusted_cost)
            
            df['adjusted_cost'] = adjusted_costs
            
            # Run standard optimization with adjusted costs
            result_df, error = self.optimize_procurement(df, budget=budget)
            
            if error:
                return pd.DataFrame(), error
            
            # Add priority information to results
            if 'equipment_type' in result_df.columns:
                result_df['priority'] = result_df['equipment_type'].map(priorities).fillna(1.0)
            
            return result_df, None
            
        except Exception as e:
            self._last_error = f"Error in priority optimization: {str(e)}"
            return pd.DataFrame(), self._last_error
    
    def sensitivity_analysis(
        self,
        equipment_data: pd.DataFrame,
        parameter: str = 'budget',
        param_range: Optional[Tuple[float, float]] = None,
        n_points: int = 10
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """
        Perform sensitivity analysis on optimization parameters.
        
        Args:
            equipment_data: DataFrame with equipment data
            parameter: Parameter to vary ('budget', 'min_demand', 'warehouse_capacity')
            param_range: Range of parameter values (min, max)
            n_points: Number of points to evaluate
            
        Returns:
            Tuple of (results DataFrame, error message)
        """
        try:
            # Set default ranges
            if param_range is None:
                if parameter == 'budget':
                    param_range = (100000, 1000000)
                elif parameter == 'min_demand':
                    param_range = (5, 50)
                elif parameter == 'warehouse_capacity':
                    param_range = (50, 200)
                else:
                    raise ValueError(f"Unknown parameter: {parameter}")
            
            # Generate parameter values
            param_values = np.linspace(param_range[0], param_range[1], n_points)
            
            results = []
            
            for val in param_values:
                kwargs = {
                    'budget': OPTIMIZATION_DEFAULTS['budget'],
                    'min_demand': int(OPTIMIZATION_DEFAULTS['min_demand']),
                    'warehouse_capacity': int(OPTIMIZATION_DEFAULTS['warehouse_capacity'])
                }
                kwargs[parameter] = val
                
                result_df, error = self.optimize_procurement(equipment_data, **kwargs)
                
                if error or result_df.empty:
                    continue
                
                total_cost = result_df['total_cost'].sum()
                total_units = result_df['optimal_quantity'].sum()
                
                results.append({
                    parameter: val,
                    'total_cost': total_cost,
                    'total_units': total_units,
                    'budget_utilization': total_cost / kwargs['budget'] * 100
                })
            
            return pd.DataFrame(results), None
            
        except Exception as e:
            self._last_error = f"Error in sensitivity analysis: {str(e)}"
            return pd.DataFrame(), self._last_error
    
    def get_optimization_summary(self) -> Dict:
        """Get summary of last optimization run."""
        return self._optimization_result or {}
    
    @property
    def last_error(self) -> Optional[str]:
        """Get last error message."""
        return self._last_error
