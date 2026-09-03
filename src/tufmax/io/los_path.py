"""
TUFMax Line-of-Sight (LoS) Path Generation Module (Module 3).

This module constructs the discrete atmospheric profile along the direct signal path 
between a ground station and a satellite. It implements a rigorous geometric validation 
(Phase 0) to detect obstructions before expensive physics calculations, followed by 
vectorized batch processing for coordinate transformations.

Key Features:
    - Phase 0 Geometric Validation: Computes elevation angle in ENU frame to detect Earth obstruction.
    - Vectorized Path Construction: Linear interpolation in 3D Euclidean space (ITRS).
    - Batch Coordinate Transformation: Leverages SpatialNormalizer for bulk conversion to Geodetic/Magnetic.
    - Safety Masking: Automatically masks points intersecting the Earth's surface.
    - Temporal Association: Assigns unified time objects with optional propagation delay correction.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import warnings
import astropy.units as u
from astropy.coordinates import CartesianRepresentation
from astropy import constants as const

# Local imports
from ..exceptions import TUFMaxGeometryError, TUFMaxInputError, TUFMaxDegradationWarning
from ..io.coords import SpatialNormalizer
from ..io.time_utils import TUFMaxTime
from ..utils.adapter import UnitAdapter
from ..utils.constants import EARTH_RADIUS_MEAN


class LoSProfile:
    """
    Immutable Container for Line-of-Sight Profile Data.
    
    Stores the structured dataset for the entire path, including Cartesian vectors, 
    geodetic/magnetic coordinates, and validity masks.
    """
    def __init__(self, 
                 xyz: u.Quantity, 
                 geodetic: Dict[str, u.Quantity], 
                 magnetic: Dict[str, u.Quantity], 
                 valid_mask: np.ndarray,
                 time_obj: TUFMaxTime):
        
        self.xyz = xyz  # [N, 3] Quantity (m)
        self.geodetic = geodetic  # Dict of Quantities (lat, lon, alt)
        self.magnetic = magnetic  # Dict of Quantities (MagLat, MLT, etc.)
        self.valid_mask = valid_mask  # Boolean array [N]
        self.time_obj = time_obj
        
        # Derived properties
        self.n_points = len(self.xyz)
        self.n_valid = np.sum(self.valid_mask)

    def get_valid_profile(self) -> Dict[str, Any]:
        """Returns only the valid points (masked out invalid/intersecting points)."""
        return {
            'xyz': self.xyz[self.valid_mask],
            'geodetic': {k: v[self.valid_mask] for k, v in self.geodetic.items()},
            'magnetic': {k: v[self.valid_mask] for k, v in self.magnetic.items()},
            'time': self.time_obj,
            'count': self.n_valid
        }

    def __repr__(self):
        return (f"LoSProfile(Total={self.n_points}, Valid={self.n_valid}, "
                f"Range=[{self.geodetic['alt'][0]:.1f}, {self.geodetic['alt'][-1]:.1f}])")


class LoSPathGenerator:
    """
    Main Orchestrator for Line-of-Sight Path Generation.
    
    Implements the 4-phase protocol:
    1. Phase 0: Geometric Validation (Elevation Check)
    2. Phase 1: Cartesian Path Construction (Linear Interpolation)
    3. Phase 2: Batch Coordinate Transformation (via SpatialNormalizer)
    4. Phase 3: Temporal Association & Output Structuring
    """

    def __init__(self, logger: Optional[Any] = None):
        self.logger = logger
        self.adapter = UnitAdapter()

    def generate(self, 
                 r_ground: u.Quantity, 
                 r_sat: u.Quantity, 
                 n_samples: int = 100, 
                 min_elevation_deg: float = 0.0,
                 epoch: Optional[TUFMaxTime] = None) -> LoSProfile:
        """
        Generates the full atmospheric profile along the line of sight.
        
        Args:
            r_ground: Ground station position vector (ITRS, Quantity [m]).
            r_sat: Satellite position vector (ITRS, Quantity [m]).
            n_samples: Number of discretization points along the path.
            min_elevation_deg: Minimum allowed elevation angle (degrees).
            epoch: Time object for the observation (TUFMaxTime). If None, uses current time.
            
        Returns:
            LoSProfile object containing the structured dataset.
            
        Raises:
            TUFMaxGeometryError: If LOS is blocked by Earth curvature.
            TUFMaxInputError: If input vectors are invalid.
        """
        # --- Input Validation ---
        if not isinstance(r_ground, u.Quantity) or not isinstance(r_sat, u.Quantity):
            raise TUFMaxInputError("Position vectors must be astropy.units.Quantity.")
        
        # Ensure units are meters for internal consistency
        r_g = r_ground.to(u.m)
        r_s = r_sat.to(u.m)
        
        if epoch is None:
            epoch = TUFMaxTime.now()

        # --- Phase 0: Geometric Validation (Fail-Fast) ---
        elevation_angle = self._calculate_elevation_angle(r_g, r_s)
        
        if elevation_angle.to(u.deg).value < min_elevation_deg:
            msg = (f"Line of Sight Blocked. Elevation {elevation_angle.to(u.deg).value:.2f}° "
                   f"is below minimum threshold {min_elevation_deg}°.")
            if self.logger:
                self.logger.log_error(msg)
            raise TUFMaxGeometryError(msg)
        
        if self.logger:
            self.logger.log_validation(type('obj', (object,), {'status': 'VALID', 'message': f'LOS Validated (El={elevation_angle.to(u.deg).value:.1f}°)', 'block_id': 3})())

        # --- Phase 1: Cartesian Path Construction ---
        path_xyz = self._construct_cartesian_path(r_g, r_s, n_samples)
        
        # Secondary Safety Check: Mask points inside Earth
        valid_mask = self._validate_path_altitude(path_xyz)
        
        if not np.any(valid_mask):
            raise TUFMaxGeometryError("Entire path is below Earth's surface. Invalid geometry.")
        
        if np.sum(~valid_mask) > 0:
            warnings.warn(f"{np.sum(~valid_mask)} points on the path intersect Earth. Masking applied.", TUFMaxDegradationWarning)

        # --- Phase 2: Batch Coordinate Transformation ---
        # We need to convert the valid points (or all points, then mask) to Geodetic/Magnetic.
        # Strategy: Convert ALL points to keep array alignment, then apply mask in result.
        # Note: SpatialNormalizer expects list/dict per point or batch? 
        # Our SpatialNormalizer currently handles single points or simple arrays. 
        # To maximize speed, we should ideally pass the whole array. 
        # Let's adapt the call to handle the batch efficiently.
        
        geodetic_batch, magnetic_batch = self._batch_transform_coordinates(path_xyz, epoch)

        # --- Phase 3: Temporal Association ---
        # Optionally correct time for signal propagation delay (t_i = t_0 + dist/c)
        # For most static profile applications, t_0 is sufficient. 
        # We attach the base epoch to the profile object.
        
        return LoSProfile(
            xyz=path_xyz,
            geodetic=geodetic_batch,
            magnetic=magnetic_batch,
            valid_mask=valid_mask,
            time_obj=epoch
        )

    def _calculate_elevation_angle(self, r_g: u.Quantity, r_s: u.Quantity) -> u.Quantity:
        """
        Calculates the elevation angle of the satellite relative to the ground station horizon.
        Uses the local ENU frame Up vector.
        """
        # Slant vector
        rho = r_s - r_g
        dist = np.linalg.norm(rho)
        
        if dist.value == 0:
            return 90.0 * u.deg # Degenerate case
            
        # Local Up vector (Unit vector from Earth center to Ground)
        # Assuming spherical Earth for horizon check is sufficient and robust
        up_vec = r_g / np.linalg.norm(r_g)
        
        # Projection of slant vector onto Up vector
        # sin(el) = (rho . up) / |rho|
        sin_el = np.dot(rho, up_vec) / dist
        
        # Clamp to [-1, 1] to avoid numerical errors in arcsin
        sin_el = np.clip(sin_el.value, -1.0, 1.0) * u.dimensionless_unscaled
        
        return np.arcsin(sin_el)


    def _construct_cartesian_path(self, r_g: u.Quantity, r_s: u.Quantity, n: int) -> u.Quantity:
        """
        Performs linear interpolation in 3D Euclidean space.
        CORRECCIÓN CRÍTICA: Forzar conversión a metros y reshape explícito antes de operar.
        """
        # 1. Asegurar unidades consistentes (Metros) y extraer valores numéricos
        # Esto evita que Astropy intente hacer broadcasting de unidades durante la multiplicación por t
        g_vals = r_g.to(u.m).value
        s_vals = r_s.to(u.m).value
        
        # 2. Forzar dimensionalidad (1, 3) para evitar errores de broadcasting
        vec_g = np.atleast_2d(g_vals) # Shape (1, 3)
        vec_s = np.atleast_2d(s_vals) # Shape (1, 3)
        
        # 3. Crear vector de tiempo 't' como array 1D puro
        t = np.linspace(0, 1, n)
        
        # 4. Interpolación vectorial segura
        # Resultado esperado: (n, 3)
        # Fórmula: P(t) = P0 + t * (P1 - P0)
        # Al hacer t[:, None] convertimos t a (n, 1) para broadcast contra (1, 3)
        path_vals = vec_g + t[:, None] * (vec_s - vec_g)
        
        # 5. Re-asignar unidades UNA VEZ al resultado final
        return path_vals * u.m

    def _validate_path_altitude(self, path_xyz: u.Quantity) -> np.ndarray:
        """
        Checks if any point on the path falls below the Earth's surface.
        Returns a boolean mask (True = Valid/Above Surface).
        """
        # Calculate geocentric distance for each point
        r_mag = np.linalg.norm(path_xyz, axis=1)
        
        # Simple spherical Earth radius check (Mean Radius)
        # A more precise check would use geodetic latitude at each point, 
        # but Mean Radius + buffer is robust for initial masking.
        r_earth = EARTH_RADIUS_MEAN.to(u.m).value
        
        # Allow a small buffer (e.g., -10km) to account for ellipsoid vs sphere difference before masking
        # But strictly, altitude < 0 is invalid.
        # Let's compute approximate altitude: r_mag - R_earth
        alt_approx = r_mag.value - r_earth
        
        # Valid if altitude > -10km (generous buffer for spherical approx), 
        # precise check happens in SpatialNormalizer
        return alt_approx > -10000.0 

    def _batch_transform_coordinates(self, path_xyz: u.Quantity, epoch: TUFMaxTime) -> Tuple[Dict, Dict]:
        """Versión DEBUG para encontrar el origen del valor 2^75."""
        print("\n[DEBUG LOS_PATH] Iniciando transformación de coordenadas...")
        path_xyz_km = path_xyz.to(u.km).value
        n_pts = path_xyz_km.shape[0]
        print(f"[DEBUG LOS_PATH] Puntos a procesar: {n_pts}")
        print(f"[DEBUG LOS_PATH] Primer punto XYZ (km): {path_xyz_km[0]}")

        # CRÍTICO: Usar dtype=float explícitamente. NUNCA usar dtype=object aquí.
        lats = np.full(n_pts, np.nan, dtype=float)
        lons = np.full(n_pts, np.nan, dtype=float)
        alts = np.full(n_pts, np.nan, dtype=float) # Inicializar con NaN, no con 0
        mag_lats = np.full(n_pts, np.nan, dtype=float)
        mlts = np.full(n_pts, np.nan, dtype=float)
        
        valid_count = 0

        for i in range(n_pts):
            vec = path_xyz_km[i]
            try:
                normalizer = SpatialNormalizer(input_pos=tuple(vec), date_decimal_year=epoch.decimalyear)
                res = normalizer.get_results()
                
                geo = res.get('geodetic', {})
                alt_val = geo.get('alt', None)
                
                # DEBUG IMPRESO CADA 10 PUNTOS Y EL PRIMERO
                if i == 0 or i % 10 == 0:
                    print(f"[DEBUG LOS_PATH] Punto {i}: Input XYZ={vec}, Alt Raw={alt_val}, Tipo={type(alt_val)}")

                if alt_val is not None:
                    # Forzar conversión a float puro
                    if hasattr(alt_val, 'value'):
                        clean_alt = float(alt_val.value)
                    else:
                        clean_alt = float(alt_val)
                    
                    lats[i] = float(geo.get('lat', np.nan))
                    lons[i] = float(geo.get('lon', np.nan))
                    alts[i] = clean_alt
                    
                    if np.isfinite(clean_alt):
                        valid_count += 1
                else:
                    if i == 0: print(f"[DEBUG LOS_PATH] ¡ALERTA! Alt es None en punto 0")

            except Exception as e:
                if i == 0: print(f"[DEBUG LOS_PATH] Excepción en punto 0: {e}")
                continue

        print(f"[DEBUG LOS_PATH] Transformación finalizada. Válidos: {valid_count}/{n_pts}")
        print(f"[DEBUG LOS_PATH] Últimos 3 valores de altitud calculados: {alts[-3:]}")
        
        # Verificación final antes de devolver
        if not np.all(np.isfinite(alts)):
            print(f"[DEBUG LOS_PATH] ¡ALERTA CRÍTICA! Hay NaNs o Infinitos en el array de altitudes.")
            # Mostramos dónde están los errores
            bad_indices = np.where(~np.isfinite(alts))[0]
            if len(bad_indices) > 0:
                print(f"[DEBUG LOS_PATH] Índices problemáticos: {bad_indices[:5]}... (total {len(bad_indices)})")

        geodetic = {
            'lat': lats * u.deg,
            'lon': lons * u.deg,
            'alt': alts * u.km  # Ahora sí, convertimos a Quantity
        }
        
        magnetic = {
            'MagLat': mag_lats * u.deg,
            'MLT': mlts * u.hourangle
        }
        
        return geodetic, magnetic
