"""
TUFMax Geometry Stress Test.
Valida robustez geométrica antes de implementar física compleja.
"""
import sys
import os
import numpy as np
from astropy import units as u
from datetime import datetime, timedelta

script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, 'src')
if src_path not in sys.path:
    sys.path.append(src_path) 

from tufmax.core.engine import TUFMaxEngine
from tufmax.exceptions import TUFMaxGeometryError

# TLE ISS (Época ~ Ago 2026)
TLE_ISS = [
    "1 25544U 98067A   26229.79251732  .00005860  00000-0  11255-3 0  9993",
    "2 25544  51.6334 355.1923 0007535  57.5442 302.6274 15.49477092581308"
]

def run_test_case(name, ground_coords, target_time, min_el=5.0):
    print(f"\n{'='*60}")
    print(f"🧪 PRUEBA: {name}")
    print(f"   Ground: {ground_coords} | Time: {target_time}")
    print(f"{'='*60}")
    
    config = {
        'min_elevation': min_el,
        'dry_run': False,
        'ground_pos': {'lat': ground_coords[0], 'lon': ground_coords[1], 'alt': 0.0},
        'sat_pos': {'lat': 0, 'lon': 0, 'alt': 400000}
    }
    
    engine = TUFMaxEngine(config=config)
    
    try:
        # Forzamos usar SGP4 raw para probar el fallback (opcional, pon True para usar Skyfield)
        # Nota: Para esta prueba, asumimos que ya arreglaste orbit.py para que SGP4 funcione bien.
        result = engine.run(tle_data=TLE_ISS, ground_coords=ground_coords, target_time=target_time, n_samples=50)
        
        if result.los_profile is None:
            print("   ❌ FALLÓ: No se generó los_profile (posible bloqueo por elevación).")
            return False

        # 1. Verificación de Longitud vs Altitud
        r_start = result.los_profile.xyz[0]
        r_end = result.los_profile.xyz[-1]
        path_len = np.linalg.norm(r_end - r_start).to(u.km).value
        
        h_start = result.los_profile.geodetic['alt'][0].to(u.km).value
        h_end = result.los_profile.geodetic['alt'][-1].to(u.km).value
        diff_h = abs(h_end - h_start)
        
        ratio = path_len / diff_h if diff_h > 0 else float('inf')
        
        print(f"   📏 Path Length: {path_len:.2f} km")
        print(f"   📏 Diff Alt:   {diff_h:.2f} km")
        print(f"   📏 Ratio:     {ratio:.2f}x")
        
        # 2. Verificación de Uniformidad del Muestreo
        diffs = []
        for i in range(len(result.los_profile.xyz) - 1):
            d = np.linalg.norm(result.los_profile.xyz[i+1] - result.los_profile.xyz[i]).to(u.km).value
            diffs.append(d)
        
        std_dev = np.std(diffs)
        mean_step = np.mean(diffs)
        print(f"   📊 Step Mean: {mean_step:.2f} km | Std Dev: {std_dev:.4f} km")
        
        if std_dev > 0.1: # Si la desviación es > 100m, algo raro pasa en la interpolación
            print("   ⚠️  ALERTA: Muestreo no uniforme en el path.")
            
        # 3. Coordenadas Magnéticas
        mag_lat = result.los_profile.magnetic.get('MagLat', None)
        if mag_lat is not None:
            valid_mag = np.all(np.isfinite(mag_lat.value))
            print(f"   🧲 MagLat Valid: {valid_mag} (Range: {mag_lat[0]:.1f} to {mag_lat[-1]:.1f})")
            if not valid_mag:
                print("   ❌ FALLÓ: Coordenadas magnéticas inválidas (NaN/Inf).")
                return False

        print("   ✅ PASÓ")
        return True

    except Exception as e:
        print(f"   ❌ ERROR: {type(e).__name__}: {e}")
        return False

def main():
    print("🚀 Iniciando TUFMax Geometry Stress Test...")
    
    results = []
    
    # 1. Caso Normal (Elevación Alta)
    results.append(run_test_case(
        "Paso Cenital Aproximado", 
        ground_coords=(51.65, 136.39, 0.0), 
        target_time="2026-08-19 07:01:16"
    ))
    
    # 2. Cambio de Día (Midnight Crossing)
    results.append(run_test_case(
        "Cruce de Medianoche", 
        ground_coords=(0.0, 0.0, 0.0), 
        target_time="2026-08-19 23:59:00"
    ))
    
    results.append(run_test_case(
        "Inicio Día Siguiente", 
        ground_coords=(0.0, 0.0, 0.0), 
        target_time="2026-08-20 00:01:00"
    ))
    
    # 3. Latitud Extrema (Casi Polo) - Validar coordenadas magnéticas
    results.append(run_test_case(
        "Estación Polar (Norte)", 
        ground_coords=(85.0, 0.0, 0.0), 
        target_time="2026-08-19 12:00:00"
    ))
    
    # 4. Baja Elevación (Forzar límite)
    # Nota: Encontrar un tiempo exacto de baja elevación requiere efemérides, 
    # pero probamos con una ubicación donde el ISS esté bajo.
    results.append(run_test_case(
        "Posible Baja Elevación", 
        ground_coords=(-35.0, 300.0, 0.0), 
        target_time="2026-08-19 05:00:00",
        min_el=0.0 # Permitir todo para ver qué pasa
    ))

    # Resumen
    print("\n" + "="*60)
    print("📊 RESUMEN DE VALIDACIÓN")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Pruebas Pasadas: {passed}/{total}")
    
    if passed == total:
        print("🎉 ¡GEOMETRÍA ROBUSTA! Listo para implementar Física.")
    else:
        print("⚠️  Hay inconsistencias. Revisar antes de continuar.")

if __name__ == "__main__":
    main()
    
# Antes de codificar los módulos de física (Bloques 4-10), debes realizar una Batería de Pruebas de Estrés Geométrico y Temporal. Si la geometría falla en casos límite, los modelos físicos (que dependen críticamente de la altitud y coordenadas magnéticas) producirán resultados erróneos o "NaN".
# Aquí tienes la lista de verificación crítica (Checklist) y el script para validarla:
# 🛡️ Checklist de Robustez Pre-Física

    # Consistencia de Propagadores (Skyfield vs SGP4):
        # Objetivo: Asegurar que el fallback a SGP4 raw no introduzca errores > 1-2 km.
        # Prueba: Ejecutar con SKYFIELD_AVAILABLE = True y luego False. Comparar posiciones ITRS finales.
        # Criterio de Aceptación: Diferencia < 5 km en cualquier instante.
    # Geometría de Baja Elevación (Horizonte):
        # Objetivo: Verificar que el cálculo de elevación sea preciso cerca del límite (min_elevation).
        # Prueba: Buscar un tiempo donde el satélite pase cerca del horizonte (El ~5°-10°).
        # Criterio de Aceptación: El path debe ser mucho más largo que la diferencia de altitud (factor > 5x o 10x). Si el factor es ~1.0 cerca del horizonte, hay un error grave.
    # Continuidad Temporal (Salto de Día/Año):
        # Objetivo: Detectar errores en el cálculo de tsince o JD cerca de medianoche o fin de año.
        # Prueba: Ejecutar en 23:59:00 y 00:01:00 del día siguiente.
        # Criterio de Aceptación: La trayectoria del satélite debe ser continua (sin saltos bruscos de posición).
    # Validación de Coordenadas Magnéticas (IGRF/World Magnetic Model):
        # Objetivo: Asegurar que SpatialNormalizer no devuelva NaN o valores basura en latitudes extremas o anomalías.
        # Prueba: Probar estaciones en polos (Lat ~90) y en la Anomalía del Atlántico Sur.
        # Criterio de Aceptación: MagLat y MLT deben ser valores finitos y razonables.
    # Interpolación de Path (Muestreo):
        # Objetivo: Confirmar que n_samples distribuye puntos uniformemente en la cuerda 3D.
        # Prueba: Calcular la distancia entre puntos consecutivos del path.
        # Criterio de Aceptación: La desviación estándar de las distancias inter-punto debe ser casi cero.

# 🧪 Script de Validación Integral (test_geometry_stress.py)
# Copia y ejecuta este script para automatizar las pruebas anteriores. Este script forzará al motor a pasar por varios escenarios críticos.

# ¿Qué hacer si fallan las pruebas?

    # Si falla "Cruce de Medianoche": El error está en orbit.py dentro del cálculo de jd_now o tsince. Revisa cómo manejas el cambio de día en datetime o el cálculo de Julian Date.
    # Si falla "Estación Polar": El problema suele estar en coords.py (SpatialNormalizer) al calcular la latitud magnética cerca de los polos (singularidad de coordenadas) o en el modelo IGRF si no maneja bien esas latitudes.
    # Si el "Ratio" es ~1.0 en baja elevación: Significa que el vector del satélite y el de la estación están casi alineados verticalmente en tu cálculo, lo cual es físicamente imposible si la elevación es baja. Revisa la rotación TEME->ITRS nuevamente.

# Una vez que este script pase con ✅ en todos los casos, puedes proceder con confianza a implementar los bloques de física (NRLMSISE-00, IRI, etc.), sabiendo que las entradas (altitud, latitud, longitud, tiempo) son correctas.
