"""
TUFMax Temporal Normalization Module (Module 1).

This module implements a hierarchical temporal normalization protocol to resolve the 
heterogeneity of time scales used by different atmospheric and geomagnetic models.
It establishes astropy.time.Time as the canonical internal representation while 
providing a robust fallback to Python's standard datetime library for restricted environments.

Key Features:
    - Multi-format Parsing: Accepts ISO strings, component lists, datetime objects, and decimal years.
    - Hierarchical Fallback: Gold Standard (Astropy) -> Legacy (Datetime + Manual Calc).
    - Derivative Computation: Automatically calculates formats required by pymsis, iri2016, SpacePy.
    - Leap Second Awareness: Handled automatically by Astropy; warned upon in Legacy mode.
"""

from __future__ import annotations
from typing import Union, List, Tuple, Dict, Any, Optional
from datetime import datetime, timezone
import warnings
import math

# Try to import Astropy (Gold Standard)
try:
    from astropy.time import Time as AstroTime
    from astropy.utils.exceptions import AstropyWarning
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False

# Local imports for consistency
from ..exceptions import TUFMaxInputError, TUFMaxDegradationWarning


class TUFMaxTime:
    """
    Robust Temporal Adapter for TUFMax.
    
    This class encapsulates a point in time, normalizing various input formats into 
    a single canonical representation. It handles scale conversions (UTC, TAI, TT) 
    and provides specific derivatives required by downstream physics models.
    
    Attributes:
        _canonical: The internal time object (Astropy Time or datetime).
        _mode: 'ASTROPY' or 'LEGACY'.
        _warnings_issued: List of warnings generated during initialization.
    """

    def __init__(self, time_input: Union[str, List[int], Tuple[int], datetime, float, 'TUFMaxTime']):
        """
        Initialize TUFMaxTime with robust parsing and fallback logic.
        
        Args:
            time_input: Raw temporal data. Supported formats:
                - ISO 8601 string (e.g., "2023-10-27T14:30:00")
                - List/Tuple of components [Y, M, D, h, m, s]
                - datetime object
                - Decimal year (float, e.g., 2023.85)
                - Another TUFMaxTime instance (copy constructor)
                
        Raises:
            TUFMaxInputError: If the input format is completely unrecognized.
        """
        self._mode = 'LEGACY'
        self._canonical: Optional[Union[AstroTime, datetime]] = None
        self._decimal_year_manual: Optional[float] = None
        
        # Copy constructor
        if isinstance(time_input, TUFMaxTime):
            self._canonical = time_input._canonical
            self._mode = time_input._mode
            self._decimal_year_manual = time_input._decimal_year_manual
            return

        # Phase 1: Robust Parsing via Astropy (Gold Standard)
        if ASTROPY_AVAILABLE:
            try:
                self._initialize_astropy(time_input)
                return
            except (ValueError, TypeError, IndexError) as e:
                # If Astropy fails to parse, we log the attempt but proceed to fallback
                # unless the input is clearly invalid for any parser.
                pass
        
        # Phase 2: Manual Fallback (Restricted Environment or Astropy Failure)
        self._initialize_legacy(time_input)

    def _initialize_astropy(self, time_input: Any):
        """Attempt to parse input using Astropy Time."""
        t_obj = None
        
        # Heuristic to detect decimal year float early to avoid ISO parsing errors
        if isinstance(time_input, float) and 1900 < time_input < 2100:
             # Explicitly tell Astropy it's a decimal year
             t_obj = AstroTime(time_input, format='decimalyear', scale='utc')
        elif isinstance(time_input, str):
            # Let Astropy infer format first (robust for ISO)
            try:
                t_obj = AstroTime(time_input)
            except ValueError:
                # Retry with explicit ISO if inference fails
                t_obj = AstroTime(time_input, format='iso', scale='utc')
        elif isinstance(time_input, (list, tuple)):
            if len(time_input) == 6:
                Y, M, D, h, m, s = time_input
                iso_str = f"{int(Y):04d}-{int(M):02d}-{int(D):02d}T{int(h):02d}:{int(m):02d}:{int(s):02d}"
                t_obj = AstroTime(iso_str, format='iso', scale='utc')
            else:
                raise ValueError("List/Tuple must have 6 components [Y, M, D, h, m, s]")
        elif isinstance(time_input, datetime):
            t_obj = AstroTime(time_input, format='datetime', scale='utc')
        else:
            raise ValueError(f"Unsupported input type for Astropy: {type(time_input)}")

        if t_obj is not None:
            self._canonical = t_obj
            self._mode = 'ASTROPY'

    def _initialize_legacy(self, time_input: Any):
        """Fallback parsing using standard datetime and manual calculations."""
        self._mode = 'LEGACY'
        dt_obj = None
        
        # Issue warning once about degraded precision
        warnings.warn(
            "TUFMaxTime: Astropy unavailable or failed. Using Legacy datetime mode. "
            "Leap seconds and complex scale conversions are NOT handled.",
            TUFMaxDegradationWarning
        )

        if isinstance(time_input, datetime):
            dt_obj = time_input
        elif isinstance(time_input, str):
            # Try ISO format manually
            try:
                # Handle 'Z' suffix for UTC
                clean_str = time_input.replace('Z', '+00:00')
                if '+' in clean_str or '-' in clean_str[10:]:
                     # Python 3.7+ fromisoformat handles timezone
                     dt_obj = datetime.fromisoformat(clean_str)
                else:
                     dt_obj = datetime.fromisoformat(clean_str)
                
                # Ensure UTC if naive
                if dt_obj.tzinfo is None:
                    dt_obj = dt_obj.replace(tzinfo=timezone.utc)
            except ValueError:
                raise TUFMaxInputError(f"Unrecognized date string format: {time_input}")
                
        elif isinstance(time_input, (list, tuple)):
            if len(time_input) == 6:
                Y, M, D, h, m, s = [int(x) for x in time_input]
                dt_obj = datetime(Y, M, D, h, m, s, tzinfo=timezone.utc)
            else:
                raise TUFMaxInputError("List/Tuple must have 6 components [Y, M, D, h, m, s]")
        elif isinstance(time_input, float):
            # Manual Decimal Year -> Datetime conversion
            Y = int(time_input)
            frac = time_input - Y
            # Approximate days in year (ignoring leap seconds/details for legacy)
            days_in_year = 366 if (Y % 4 == 0 and (Y % 100 != 0 or Y % 400 == 0)) else 365
            day_of_year = int(frac * days_in_year) + 1
            
            # Reconstruct date from DOY (simplified)
            start_of_year = datetime(Y, 1, 1, tzinfo=timezone.utc)
            from datetime import timedelta
            dt_obj = start_of_year + timedelta(days=day_of_year - 1)
            
            # Store the original decimal year for direct access
            self._decimal_year_manual = time_input
        else:
            raise TUFMaxInputError(f"Unrecognized temporal format: {type(time_input)}")

        if dt_obj:
            self._canonical = dt_obj
            # If we didn't set decimal year manually (from float input), calculate it now
            if self._decimal_year_manual is None:
                self._decimal_year_manual = self._calculate_manual_decimal_year(dt_obj)

    def _calculate_manual_decimal_year(self, dt: datetime) -> float:
        """
        Manually compute decimal year: Y_dec = Y + (DOY - 1) / 365.2425
        Formula from pseudocode Algorithm 1.
        """
        start_of_year = datetime(dt.year, 1, 1, tzinfo=timezone.utc)
        delta = dt - start_of_year
        doy = delta.days + 1 # 1-based day of year
        
        # Use standard tropical year length as per pseudocode
        return float(dt.year) + (doy - 1) / 365.2425

    # ----------------------------------------------------------------------
    # Derivative Computation for Legacy Libraries
    # ----------------------------------------------------------------------

    @property
    def datetime(self) -> datetime:
        """
        Returns standard datetime object.
        Required for: pymsis
        """
        if self._mode == 'ASTROPY':
            return self._canonical.datetime # type: ignore
        return self._canonical # type: ignore

    @property
    def decimalyear(self) -> float:
        """
        Returns decimal year float.
        Required for: iri2016
        """
        if self._mode == 'ASTROPY':
            return self._canonical.decimalyear # type: ignore
        if self._decimal_year_manual is not None:
            return self._decimal_year_manual
        # Fallback calculation if somehow missing
        return self._calculate_manual_decimal_year(self._canonical) # type: ignore

    @property
    def mjd(self) -> float:
        """
        Returns Modified Julian Date.
        Required for: spacepy
        """
        if self._mode == 'ASTROPY':
            return self._canonical.mjd # type: ignore
        
        # Manual MJD calc for Legacy (MJD = JD - 2400000.5)
        # Simplified approximation for legacy mode
        dt = self._canonical # type: ignore
        # Gregorian to JD algorithm
        Y = dt.year
        M = dt.month
        D = dt.day + dt.hour/24.0 + dt.minute/1440.0 + dt.second/86400.0
        
        if M <= 2:
            Y -= 1
            M += 12
            
        A = Y // 100
        B = 2 - A + (A // 4)
        
        JD = int(365.25 * (Y + 4716)) + int(30.6001 * (M + 1)) + D + B - 1524.5
        return JD - 2400000.5

    @property
    def unix(self) -> float:
        """
        Returns Unix Timestamp (seconds since 1970-01-01 UTC).
        Required for: spacepy, general logging
        """
        if self._mode == 'ASTROPY':
            return self._canonical.unix # type: ignore
        
        dt = self._canonical # type: ignore
        return dt.timestamp()

    @property
    def iso(self) -> str:
        """Returns ISO 8601 string representation."""
        if self._mode == 'ASTROPY':
            return self._canonical.iso # type: ignore
        return self._canonical.isoformat() # type: ignore

    @property
    def mode(self) -> str:
        """Returns the operational mode ('ASTROPY' or 'LEGACY')."""
        return self._mode

    def __repr__(self):
        return f"TUFMaxTime({self.iso}, mode={self._mode})"

    # ----------------------------------------------------------------------
    # Static Helpers
    # ----------------------------------------------------------------------

    @staticmethod
    def now() -> 'TUFMaxTime':
        """Creates a TUFMaxTime instance for the current UTC moment."""
        return TUFMaxTime(datetime.now(timezone.utc))

    @staticmethod
    def parse_any(input_val: Any) -> 'TUFMaxTime':
        """Factory method to explicitly parse any supported input."""
        return TUFMaxTime(input_val)
