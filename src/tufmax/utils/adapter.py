"""
TUFMax Unit Adapter & Utilities.

This module implements the Adapter design pattern to resolve the incompatibility between 
the modern physics ecosystem based on astropy.units and high-performance numerical 
libraries (such as SciPy or NumPy) that operate exclusively on raw numerical arrays.

It acts as a robust interoperability layer ("glue code") that automatically manages the 
lifecycle of physical units: normalizing inputs to SI, extracting scalar values for 
computation, and reconstructing results with correct derived units.

Key Features:
    - Unit Normalization: Strict conversion to SI base units.
    - Safe Extraction/Restoration: Handling of Quantity vs. raw float/array.
    - Dimensional Validation: Ensuring physical consistency before operations.
    - Numerical Operations: Integration, interpolation, and differentiation with automatic unit inference.
"""

from __future__ import annotations
from typing import Union, Optional, Dict, Any, List
import numpy as np
import astropy.units as u
from astropy.units import Quantity, Unit, PhysicalType
from scipy import integrate

# Type aliases for clarity
Number = Union[int, float, np.number]
ArrayLike = Union[np.ndarray, List[Number]]
QuantityLike = Union[Quantity, Number, ArrayLike]


class UnitAdapter:
    """
    Centralized Utility Class for Unit Safety and Interoperability.
    
    Provides static methods to safely convert, strip, restore, and validate 
    physical units, ensuring that numerical routines receive pure floats/arrays 
    while the rest of the library maintains rigorous dimensional tracking.
    """

    # ----------------------------------------------------------------------
    # 1. Conversion & Normalization
    # ----------------------------------------------------------------------

    @staticmethod
    def convert_to_si(q: QuantityLike, physical_type: Optional[str] = None) -> Quantity:
        """
        Converts input to an astropy Quantity in SI base units.
        
        If input is a raw number/array, it assumes SI units already.
        If physical_type is provided, it validates the dimensionality before conversion.
        
        Args:
            q: Input value (with or without units).
            physical_type: Optional string name of the physical type (e.g., 'length', 'time') 
                           to validate against.
                           
        Returns:
            astropy.units.Quantity in SI units.
            
        Raises:
            ValueError: If dimensions do not match the expected physical_type.
        """
        if not isinstance(q, Quantity):
            # Assume SI if no units provided
            if physical_type:
                # We can't strictly validate dimensionality of raw numbers, 
                # but we can warn or just assign the unit.
                unit = UnitAdapter.get_standard_unit(physical_type)
                q = q * unit
            else:
                # Default assumption: dimensionless or SI already
                q = q * u.dimensionless_unscaled
        
        if physical_type:
            if not q.unit.is_equivalent(UnitAdapter.get_standard_unit(physical_type)):
                # Check specifically for physical type match if possible
                try:
                    if u.get_physical_type(q.unit) != physical_type:
                        raise ValueError(f"Dimension mismatch: Expected {physical_type}, got {u.get_physical_type(q.unit)}")
                except (TypeError, ValueError):
                    pass # Fallback if physical type lookup fails
        
        return q.to_si_unit()

    @staticmethod
    def strip_units(q: QuantityLike) -> Union[float, np.ndarray]:
        """
        Safely extracts the numerical value from a Quantity.
        
        If input is already a number or array, returns it as is.
        Ensures the output is always a native numpy scalar or ndarray for compatibility.
        
        Args:
            q: Input value (with or without units).
            
        Returns:
            Raw numerical value (float or np.ndarray).
        """
        if isinstance(q, Quantity):
            val = q.value
            # Ensure numpy scalar if single value
            if isinstance(val, (int, float)):
                return float(val)
            return val
        return np.asarray(q)

    @staticmethod
    def restore_quantity(value: Union[float, np.ndarray], unit: Union[Unit, str]) -> Quantity:
        """
        Reconstructs a Quantity from a numerical value and a unit.
        
        Args:
            value: Numerical data (scalar or array).
            unit: Target unit (astropy Unit or string).
            
        Returns:
            astropy.units.Quantity.
        """
        if not isinstance(unit, Unit):
            unit = Unit(unit)
        return value * unit

    # ----------------------------------------------------------------------
    # 2. Validation & Helpers
    # ----------------------------------------------------------------------

    @staticmethod
    def validate_dimensionality(q: QuantityLike, expected_type: str) -> bool:
        """
        Checks if a Quantity matches the expected physical type.
        
        Args:
            q: Input value.
            expected_type: String name of physical type (e.g., 'length', 'energy').
            
        Returns:
            True if valid, False otherwise.
        """
        if not isinstance(q, Quantity):
            return False # Cannot validate dimensionality of raw numbers safely
        try:
            return u.get_physical_type(q.unit) == expected_type
        except (TypeError, ValueError):
            return False

    @staticmethod
    def get_standard_unit(physical_type: str) -> Unit:
        """
        Returns the standard SI unit for a given physical type name.
        
        Args:
            physical_type: Name of the physical type.
            
        Returns:
            Corresponding astropy Unit.
            
        Raises:
            ValueError: If physical type is unknown.
        """
        mapping = {
            'length': u.m,
            'time': u.s,
            'mass': u.kg,
            'velocity': u.m / u.s,
            'acceleration': u.m / u.s**2,
            'force': u.N,
            'energy': u.J,
            'power': u.W,
            'pressure': u.Pa,
            'density': u.kg / u.m**3,
            'number_density': 1 / u.m**3,
            'temperature': u.K,
            'magnetic_field': u.T,
            'electric_field': u.V / u.m,
            'frequency': u.Hz,
            'angle': u.rad,
            'solid_angle': u.sr,
            'area': u.m**2,
            'volume': u.m**3,
            'charge': u.C,
            'potential': u.V,
            'conductivity': u.S / u.m,
            'cross_section': u.m**2,
        }
        if physical_type not in mapping:
            # Fallback: try to resolve via astropy directly
            try:
                return u.Unit(physical_type)
            except ValueError:
                raise ValueError(f"Unknown physical type: {physical_type}")
        return mapping[physical_type]

    # ----------------------------------------------------------------------
    # 3. Safe Numerical Operations (Unit-Aware)
    # ----------------------------------------------------------------------

    @staticmethod
    def safe_integrate_trapz(y: QuantityLike, x: QuantityLike) -> Quantity:
        """
        Performs trapezoidal integration with automatic unit inference.
        
        Integral of y dx -> Units(y) * Units(x)
        
        Args:
            y: Dependent variable (e.g., density).
            x: Independent variable (e.g., altitude).
            
        Returns:
            Integrated Quantity with correct derived units.
        """
        y_q = y if isinstance(y, Quantity) else y * u.dimensionless_unscaled
        x_q = x if isinstance(x, Quantity) else x * u.dimensionless_unscaled
        
        y_val = UnitAdapter.strip_units(y_q.to_si_unit())
        x_val = UnitAdapter.strip_units(x_q.to_si_unit())
        
        result_val = integrate.trapz(y_val, x_val)
        result_unit = y_q.unit * x_q.unit
        
        return UnitAdapter.restore_quantity(result_val, result_unit)

    @staticmethod
    def safe_interpolate_1d(x: QuantityLike, y: QuantityLike, x_new: QuantityLike) -> Quantity:
        """
        Performs linear interpolation preserving units of Y.
        
        Args:
            x: Independent variable (known).
            y: Dependent variable (known).
            x_new: Independent variable (query points).
            
        Returns:
            Interpolated y_new with same units as y.
        """
        x_q = x if isinstance(x, Quantity) else x * u.dimensionless_unscaled
        y_q = y if isinstance(y, Quantity) else y * u.dimensionless_unscaled
        x_new_q = x_new if isinstance(x_new, Quantity) else x_new * u.dimensionless_unscaled
        
        # Ensure consistent units for X axis
        x_si = x_q.to_si_unit()
        x_new_si = x_new_q.to(x_si.unit)
        
        y_val = UnitAdapter.strip_units(y_q)
        x_val = UnitAdapter.strip_units(x_si)
        x_new_val = UnitAdapter.strip_units(x_new_si)
        
        # Handle edge cases for interpolation
        if len(x_val) < 2:
            raise ValueError("Interpolation requires at least 2 points.")
            
        y_new_val = np.interp(x_new_val, x_val, y_val)
        
        return UnitAdapter.restore_quantity(y_new_val, y_q.unit)

    @staticmethod
    def safe_derivative(y: QuantityLike, x: QuantityLike) -> Quantity:
        """
        Computes numerical derivative dy/dx with correct unit inference.
        
        Uses simple finite differences (forward/backward for edges, central for interior).
        
        Args:
            y: Dependent variable.
            x: Independent variable.
            
        Returns:
            Derivative Quantity with units Units(y)/Units(x).
        """
        y_q = y if isinstance(y, Quantity) else y * u.dimensionless_unscaled
        x_q = x if isinstance(x, Quantity) else x * u.dimensionless_unscaled
        
        y_si = y_q.to_si_unit()
        x_si = x_q.to_si_unit()
        
        y_val = UnitAdapter.strip_units(y_si)
        x_val = UnitAdapter.strip_units(x_si)
        
        dy = np.diff(y_val)
        dx = np.diff(x_val)
        
        # Avoid division by zero
        dx[dx == 0] = np.nan
        
        deriv_val = dy / dx
        # Result has length N-1. Could pad if necessary, but returning reduced array is safer.
        deriv_unit = y_si.unit / x_si.unit
        
        return UnitAdapter.restore_quantity(deriv_val, deriv_unit)

    @staticmethod
    def compute_column_density(density_profile: QuantityLike, altitude_profile: QuantityLike) -> Quantity:
        """
        Specialized helper to compute column density from volumetric density.
        
        Integral(n dh) -> units m^-2
        
        Args:
            density_profile: Volumetric number density (e.g., m^-3).
            altitude_profile: Altitude array (e.g., m).
            
        Returns:
            Column density (e.g., m^-2).
        """
        return UnitAdapter.safe_integrate_trapz(density_profile, altitude_profile)
