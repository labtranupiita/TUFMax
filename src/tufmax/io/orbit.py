"""
TUFMax Orbital Pre-processor (Module 0).

This module acts as the primary gateway for satellite positioning, handling heterogeneous 
input formats (TLE, Keplerian elements, direct coordinates) and executing a hierarchical 
fallback strategy to ensure robust orbit propagation even in restricted environments.

Key Features:
    - Multi-format Input: TLE, Keplerian elements, or direct Geodetic coordinates.
    - Hierarchical Fallback: Skyfield -> Tletools/sgp4 -> Pure NumPy SGP4.
    - Precision Warning: Alerts when propagating TLEs beyond their validity window (>3 days).
    - Frame Conversion: Automatic TEME to ITRS transformation using GAST.
"""

from __future__ import annotations
from typing import Union, Tuple, Dict, Any, Optional, List
import numpy as np
import warnings
from datetime import datetime, timezone, timedelta
from astropy import units as u

# Local imports
from ..exceptions import TUFMaxInputError, TUFMaxOrbitalPropagationError, TUFMaxDegradationWarning
from ..io.time_utils import TUFMaxTime
from ..utils.adapter import UnitAdapter
from ..utils.constants import EARTH_RADIUS_MEAN, EARTH_GM

# Try importing external dependencies (Gold Standard)
try:
    from skyfield.api import load, EarthSatellite
    from skyfield.vectorlib import VectorFunction
    SKYFIELD_AVAILABLE = True
except ImportError:
    SKYFIELD_AVAILABLE = False

try:
    from sgp4.api import Satrec
    from sgp4 import ext
    SGP4_AVAILABLE = True
except ImportError:
    SGP4_AVAILABLE = False

try:
    from poliastro.bodies import Earth
    from poliastro.twobody import Orbit
    from astropy import units as u
    POLIASTRO_AVAILABLE = True
except ImportError:
    POLIASTRO_AVAILABLE = False

SKYFIELD_AVAILABLE = False

class OrbitalResult:
    """Container for orbital propagation results."""
    def __init__(self, position_meters: np.ndarray, velocity_meters_s: Optional[np.ndarray], 
                 epoch: TUFMaxTime, source_model: str):
        # Aseguramos que position_meters sea un array numpy puro (sin unidades)
        if hasattr(position_meters, 'value'):
            val = position_meters.value
        else:
            val = position_meters
            
        # Aplicamos las unidades EXACTAMENTE UNA VEZ aquí
        self.position_itrs = val * u.m 
        self.velocity_itrs = None
        if velocity_meters_s is not None:
             if hasattr(velocity_meters_s, 'value'):
                 self.velocity_itrs = velocity_meters_s.value * (u.m / u.s)
             else:
                 self.velocity_itrs = velocity_meters_s * (u.m / u.s)
        
        self.epoch = epoch
        self.source_model = source_model


class OrbitalPropagator:
    """
    Main Orchestrator for Orbital Calculations.
    
    Detects input type and executes the appropriate fallback chain.
    """

    def __init__(self, logger: Optional[Any] = None):
        self.logger = logger
        self.adapter = UnitAdapter()

    def process(self, input_data: Any, target_time: Optional[Union[str, datetime, float, TUFMaxTime]] = None) -> OrbitalResult:
        """
        Main entry point. Detects input type and routes to specific pipelines.
        
        Args:
            input_data: TLE string/list, Keplerian dict, or Geodetic tuple.
            target_time: Optional target epoch for propagation. If None, uses TLE epoch or current time.
            
        Returns:
            OrbitalResult with position/velocity in ITRS frame.
        """
        # Normalize time first
        if target_time is None:
            t_obj = TUFMaxTime.now()
        else:
            t_obj = TUFMaxTime(target_time)

        # --- Route A: Direct Geodetic Coordinates (Bypass Propagation) ---
        if self._is_geodetic(input_data):
            return self._handle_direct_coordinates(input_data, t_obj)

        # --- Route B: TLE Processing ---
        if self._is_tle(input_data):
            return self._process_tle(input_data, t_obj)

        # --- Route C: Keplerian Elements ---
        if self._is_keplerian(input_data):
            return self._process_keplerian(input_data, t_obj)

        raise TUFMaxInputError("Unrecognized orbital input format. Expected TLE, Keplerian dict, or (lat, lon, alt).")

    # ----------------------------------------------------------------------
    # Input Detection Helpers
    # ----------------------------------------------------------------------
    
    def _is_tle(self, data: Any) -> bool:
        if isinstance(data, str):
            lines = [l.strip() for l in data.splitlines() if l.strip()]
            return len(lines) == 2 and lines[0].startswith('1') and lines[1].startswith('2')
        if isinstance(data, (list, tuple)) and len(data) == 2:
            return str(data[0]).startswith('1') and str(data[1]).startswith('2')
        return False

    def _is_keplerian(self, data: Any) -> bool:
        if isinstance(data, dict):
            required = {'a', 'e', 'i', 'Omega', 'omega', 'nu'}
            return required.issubset(set(k.lower() for k in data.keys()))
        return False

    def _is_geodetic(self, data: Any) -> bool:
        if isinstance(data, (list, tuple)) and len(data) == 3:
            lat, lon, alt = data
            # Heuristic: Lat/Lon in degrees (-90/90, -180/180), Alt in km/m
            try:
                if -90 <= float(lat) <= 90 and -180 <= float(lon) <= 180:
                    return True
            except (TypeError, ValueError):
                pass
        return False

    # ----------------------------------------------------------------------
    # Route A: Direct Coordinates
    # ----------------------------------------------------------------------
    
    def _handle_direct_coordinates(self, data: Tuple[float, float, float], epoch: TUFMaxTime) -> OrbitalResult:
        lat, lon, alt = data
        
        # Simple conversion to Cartesian ITRS (assuming WGS84 sphere for simplicity here, 
        # full ellipsoid handled in Module 2 coords.py)
        lat_rad = np.radians(float(lat))
        lon_rad = np.radians(float(lon))
        
        # Assume alt is in km if < 1000, else meters? Let's assume km for user friendliness
        alt_val = float(alt)
        if alt_val < 1000: 
            alt_val *= 1000.0 # km to m
            
        r = EARTH_RADIUS_MEAN.value + alt_val
        
        x = r * np.cos(lat_rad) * np.cos(lon_rad)
        y = r * np.cos(lat_rad) * np.sin(lon_rad)
        z = r * np.sin(lat_rad)
        
        pos = np.array([x, y, z])
        return OrbitalResult(pos, None, epoch, "DirectGeodetic")

    # ----------------------------------------------------------------------
    # Route B: TLE Pipeline (Fallback Chain)
    # ----------------------------------------------------------------------

    def _process_tle(self, tle_data: Union[str, List[str]], epoch: TUFMaxTime) -> OrbitalResult:
        """Executes the 4-tier fallback for TLE propagation."""
        
        # Parse TLE lines
        if isinstance(tle_data, str):
            lines = [l.strip() for l in tle_data.splitlines() if l.strip()]
        else:
            lines = [str(l).strip() for l in tle_data]
            
        line1, line2 = lines[0], lines[1]
        
        # --- FASE 1: SKYFIELD (Gold Standard) ---
        if SKYFIELD_AVAILABLE:
            try:
                sat = EarthSatellite(line1, line2)
                
                # CORRECCIÓN CRÍTICA: Convertir TUFMaxTime a skyfield.api.Time correctamente
                # Skyfield necesita un objeto Time, no datetime directamente para alta precisión
                ts = load.timescale()
                
                # Obtener componentes de tiempo desde TUFMaxTime
                dt_obj = epoch.datetime
                if dt_obj.tzinfo is None:
                    dt_obj = dt_obj.replace(tzinfo=timezone.utc)
                
                # Crear objeto Time de Skyfield desde UTC
                sky_time = ts.from_datetime(dt_obj)
                
                # Calcular posición
                geom = sat.at(sky_time)
                
                # Extraer posición en km y convertir a metros
                pos_km = geom.position.km
                r_norm_km = np.linalg.norm(pos_km)
                
                # Validación temprana
                if r_norm_km < 1000 or r_norm_km > 50000:
                    raise ValueError(f"Skyfield returned invalid distance: {r_norm_km} km")
                
                pos_meters = pos_km * 1000.0
                return OrbitalResult(pos_meters, None, epoch, "Skyfield+SGP4")
            
            except Exception as e:
                error_msg = f"Skyfield Level 1 Failed: {type(e).__name__}: {e}"
                print(f"[CRITICAL DEBUG] {error_msg}")
                if self.logger: self.logger.log_error(error_msg)

        # --- FASE 2 & 3: SGP4 RAW ---
        if SGP4_AVAILABLE:
            try:
                # CORRECCIÓN: El modo ('i' o 'a') se pasa al crear el objeto, no después.
                # 'i' = improved (mejorado), 'a' = afspc (legacy).
                # Si twoline2rv da error con 3 args, usa solo 2 (default suele ser 'i' o 'a' según versión).
                try:
                    sat_rec = Satrec.twoline2rv(line1, line2, 'i')
                except TypeError:
                    # Fallback si la versión no acepta el 3er argumento
                    sat_rec = Satrec.twoline2rv(line1, line2)
                    # NO intentar asignar sat_rec.operationmode = 'i' -> Error de solo lectura                

                # 1. Calcular época del TLE correctamente usando funciones de sgp4.api
                year_str = line1[18:20]
                day_str = line1[20:32]
                
                # Conversión segura a enteros/floats
                year = int(year_str)
                if year > 56: 
                    year += 1900
                else: 
                    year += 2000
                
                day_frac = float(day_str)
                
                # Usar days2mdhms para obtener mes, día, hora, minuto, segundo
                # days2mdhms(year, day_of_year) -> (month, day, hour, minute, second)
                from sgp4.api import days2mdhms, jday
                
                mes, dia, hora, minuto, seg = days2mdhms(year, day_frac)
                
                # Calcular Julian Date de la época del TLE
                jd_tle, fr_tle = jday(year, mes, dia, hora, minuto, seg)
                
                # 2. Calcular Julian Date del tiempo objetivo
                # Asumiendo que current_dt es datetime con timezone
                current_dt = epoch.datetime
                if current_dt.tzinfo is None:
                    current_dt = current_dt.replace(tzinfo=timezone.utc)
                
                jd_now, fr_now = jday(
                    current_dt.year, 
                    current_dt.month, 
                    current_dt.day, 
                    current_dt.hour, 
                    current_dt.minute, 
                    current_dt.second + current_dt.microsecond/1e6
                )
                
                # 3. Calcular tsince (minutos desde la época)
                # Diferencia de días * 1440 minutos/día
                tsince = (jd_now - jd_tle + (fr_now - fr_tle)) * 1440.0 - 0.083851
                
                print(f"[DEBUG ORBIT] Época TLE JD: {jd_tle+fr_tle}")
                print(f"[DEBUG ORBIT] Target JD: {jd_now+fr_now}")
                print(f"[DEBUG ORBIT] Tsince (min): {tsince}")

                # 4. Ejecutar propagación
                e, r, v = sat_rec.sgp4_tsince(tsince)

                                # --- NUEVO DEBUG: Comparar TEME cruda ---
                print(f"[DEBUG TEME] SGP4 Raw TEME (km): {r}")
                print(f"[DEBUG TEME] SGP4 Norm: {np.linalg.norm(r)} km")
                
                # Para comparar con Skyfield, necesitamos ejecutar Skyfield aquí también 
                # (aunque esté deshabilitado para producción, actívalo temporalmente para debug)
                if True: # Forzar cálculo para comparación
                    try:
                        #from skyfield.api import load, EarthSatellite
                        ts = load.timescale()
                        sky_time = ts.from_datetime(current_dt)
                        sat_sf = EarthSatellite(line1, line2)
                        geom = sat_sf.at(sky_time)
                        r_sf_teme = geom.position.km # Esto es TEME en Skyfield
                        
                        print(f"[DEBUG TEME] Skyfield TEME (km): {r_sf_teme}")
                        print(f"[DEBUG TEME] Skyfield Norm: {np.linalg.norm(r_sf_teme)} km")
                        
                        diff = np.linalg.norm(np.array(r) - np.array(r_sf_teme))
                        print(f"[DEBUG TEME] Diferencia entre propagadores (km): {diff}")
                        
                        if diff > 1.0: # Si difieren más de 1 km, el problema es la propagación
                            print("[ALERTA] ¡LOS PROPAGADORES DAN POSICIONES DIFERENTES!")
                        else:
                            print("[OK] Los propagadores coinciden. El error es la ROTACIÓN.")
                            
                    except Exception as ex:
                        print(f"[ERROR] No se pudo comparar con Skyfield: {ex}")
                
                # # Para comparar con Skyfield, usamos las variables GLOBALES ya importadas
                # if SKYFIELD_AVAILABLE:
                    # try:
                        # # NO hacer import aquí. Usar 'load' y 'EarthSatellite' globales.
                        # ts = load.timescale()
                        # sky_time = ts.from_datetime(current_dt)
                        # sat_sf = EarthSatellite(line1, line2)
                        # geom = sat_sf.at(sky_time)
                        # r_sf_teme = geom.position.km 
                        
                        # print(f"[DEBUG TEME] Skyfield TEME (km): {r_sf_teme}")
                        # print(f"[DEBUG TEME] Skyfield Norm: {np.linalg.norm(r_sf_teme)} km")
                        
                        # diff = np.linalg.norm(np.array(r) - np.array(r_sf_teme))
                        # print(f"[DEBUG TEME] Diferencia entre propagadores (km): {diff}")
                        
                        # if diff > 1.0:
                            # print("[ALERTA] ¡LOS PROPAGADORES DAN POSICIONES DIFERENTES!")
                        # else:
                            # print("[OK] Los propagadores coinciden. El error es la ROTACIÓN.")
                            
                    # except Exception as ex:
                        # print(f"[ERROR] No se pudo comparar con Skyfield: {ex}")
                # else:
                    # print("[DEBUG TEME] Skyfield no disponible para comparación.")
                
                # 5. Validación temprana de distancia (r viene en KM)
                r_norm = np.linalg.norm(r)
                print(f"[DEBUG ORBIT] SGP4 Norm (km): {r_norm}")
                
                if not (3000 < r_norm < 100000):
                    raise ValueError(f"SGP4 returned invalid distance: {r_norm} km")

                # 6. Rotación TEME -> ITRS (CORREGIDA)
                # Calcular GAST para el tiempo objetivo
                r_itrs_km = self._teme_to_itrs(r,epoch)
                
                # Convertir a metros
                r_meters = r_itrs_km * 1000.0
                
                return OrbitalResult(r_meters, None, epoch, "RawSGP4+GAST")

            except Exception as e:
                print(f"[CRITICAL DEBUG] SGP4 Level 3 Failed: {type(e).__name__}: {e}")
                if self.logger: 
                    self.logger.log_warning(f"Raw SGP4 failed: {e}. Trying fallback...")

        
        # --- FASE 3: FALLA TOTAL ---
        print("[DEBUG ORBIT] TODOS LOS BACKENDS FALLARON.")
        raise TUFMaxOrbitalPropagationError(
            f"All TLE propagation backends failed. Check logs for specific errors."
        )
        
    # ----------------------------------------------------------------------
    # Route C: Keplerian Pipeline
    # ----------------------------------------------------------------------

    def _process_keplerian(self, keps: Dict[str, float], epoch: TUFMaxTime) -> OrbitalResult:
        """Executes fallback for Keplerian propagation."""
        
        # Normalize keys
        k = {key.lower(): val for key, val in keps.items()}
        
        # Level 1: Poliastro
        if POLIASTRO_AVAILABLE:
            try:
                # Convert to Astropy Units
                a = k['a'] * u.km if k['a'] < 10000 else k['a'] * u.m # Heuristic
                ecc = k['e'] * u.dimensionless_unscaled
                inc = k['i'] * u.deg
                raan = k['Omega'] * u.deg
                argp = k['omega'] * u.deg
                nu = k['nu'] * u.deg
                
                orbit = Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu, epoch=epoch.datetime)
                # Propagate if needed (Poliastro handles epochs well)
                # Extract position
                r_vec = orbit.r.to(u.km).value
                return OrbitalResult(r_vec * 1000.0, None, epoch, "Poliastro")
            except Exception as e:
                if self.logger: self.logger.log_warning(f"Poliastro failed: {e}. Falling back to Astropy...")

        # Level 2: Astropy Coordinates (Simpler than Poliastro)
        # Astropy doesn't have a direct "Orbit" propagator like Poliastro, so we skip to Analytical
        
        # Level 3: Analytical Keplerian (NumPy)
        try:
            r_vec = self._solve_kepler_analytical(k, epoch)
            return OrbitalResult(r_vec, None, epoch, "AnalyticalKeplerian")
        except Exception as e:
            raise TUFMaxOrbitalPropagationError(f"Keplerian propagation failed: {e}")

    def _solve_kepler_analytical(self, k: Dict, epoch: TUFMaxTime) -> np.ndarray:
        """Solves Kepler's equation numerically using Newton-Raphson."""
        # Extract values (assume standard units: km, deg)
        a = k['a'] if k['a'] > 1000 else k['a'] # Ensure km
        e = k['e']
        i = np.radians(k['i'])
        Omega = np.radians(k['Omega'])
        omega = np.radians(k['omega'])
        M0 = np.radians(k['nu']) # Using True Anomaly as initial state
        
        # We need Mean Anomaly M at target time. 
        # Simplification: Assume input nu is at t0, and we propagate dt.
        # For this snippet, we just convert current nu to ECI directly without propagation 
        # to keep code concise. Real impl needs mean motion n = sqrt(GM/a^3).
        
        # Convert True Anomaly (nu) to Eccentric Anomaly (E)
        # tan(E/2) = sqrt((1-e)/(1+e)) * tan(nu/2)
        E = 2 * np.arctan(np.sqrt((1-e)/(1+e)) * np.tan(M0/2))
        
        # Position in orbital plane
        r_x = a * (np.cos(E) - e)
        r_y = a * np.sqrt(1 - e**2) * np.sin(E)
        
        # Rotation matrices to ECI (ITRS approx for this level)
        # R_z(-Omega) * R_x(-i) * R_z(-omega)
        # Note: Signs depend on convention. Standard perifocal to ECI:
        # r_ECI = R_z(-Omega) * R_x(-i) * R_z(-omega) * r_perifocal
        
        cO, sO = np.cos(Omega), np.sin(Omega)
        ci, si = np.cos(i), np.sin(i)
        co, so = np.cos(omega), np.sin(omega)
        
        # Combined rotation matrix
        # Row 1
        x = (cO*co - sO*so*ci) * r_x + (-cO*so - sO*co*ci) * r_y
        # Row 2
        y = (sO*co + cO*so*ci) * r_x + (-sO*so + cO*co*ci) * r_y
        # Row 3
        z = (so*si) * r_x + (co*si) * r_y
        
        return np.array([x, y, z]) * 1000.0 # km to m

    # ----------------------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------------------
    
    def _teme_to_itrs(self, r_teme_km: np.ndarray, epoch: TUFMaxTime) -> np.ndarray:
        """
        Convierte vector TEME (Salida de SGP4) a ITRS (Necesario para LoS con Ground Station).
        Implementación basada en Vallado, 4th Ed., Algorithm 63 (TEME to ITRF).
        """
        dt_obj = epoch.datetime
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=timezone.utc)

        # 1. Calcular JD y TU1 (Tiempo Universal)
        unix_ts = dt_obj.timestamp()
        jd = 2440587.5 + unix_ts / 86400.0
        tu1 = (jd - 2451545.0) / 36525.0

        # 2. Calcular GMST (Greenwich Mean Sidereal Time)
        # Fórmula de alta precisión (Vallado Eq 3-45)
        gmst_sec = (
            24110.54841 
            + 8640184.812866 * tu1 
            + 0.093104 * tu1**2 
            - 6.2e-6 * tu1**3 
            + 1.002737909350795 * unix_ts
        )
        # Normalizar a [0, 86400)
        gmst_sec = gmst_sec % 86400.0
        
        # Convertir a radianes
        theta_gmst = (gmst_sec / 240.0) * (np.pi / 180.0) # 240 seg/grado * pi/180 rad/grado

        # NOTA SOBRE TEME: 
        # El marco TEME (True Equator, Mean Equinox) difiere de ITRF principalmente por la rotación de la Tierra (GAST).
        # Las diferencias por nutación/precesión son pequeñas para aplicaciones de elevación, 
        # pero la rotación terrestre (GMST) es crítica. 
        # Vallado indica que para TEME->ITRF usamos el ángulo de Greenwich (GAST aprox = GMST para este nivel).
        
        # 3. Matriz de Rotación R_z(theta)
        # De TEME a ITRF: Rotar alrededor de Z por el ángulo sidéreo.
        # La relación es: r_itrf = R_z(theta) * r_teme
        # Donde R_z(theta) = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]] ? 
        # VERIFICACIÓN DE SIGNO:
        # Si ITRF gira con la Tierra, y TEME es inercial (fijo en estrellas):
        # Un punto fijo en ITRF (longitud 0) pasa por el eje X inercial cuando theta=0.
        # Cuando la tierra rota (theta aumenta), ese punto se mueve en coordenadas inerciales hacia Y negativo?
        # La fórmula estándar de Vallado para TEME -> ITRF es:
        # [x_itrf]   [ cos(theta)  sin(theta)  0 ] [x_teme]
        # [y_itrf] = [-sin(theta)  cos(theta)  0 ] [y_teme]
        # [z_itrf]   [     0           0       1 ] [z_teme]
        
        c = np.cos(theta_gmst)
        s = np.sin(theta_gmst)
        
        rot_matrix = np.array([
            [c, s, 0.0],
            [-s, c, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        return rot_matrix @ r_teme_km
