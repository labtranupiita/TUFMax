"""
TUFMax Hello World Test.
Valida la cadena de ingesta, geometría y orquestación del motor.
"""

import sys
import os
from astropy import units as u

# Obtener la ruta absoluta al directorio 'src'
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, 'src')

# Agregar la ruta solo si no está ya presente (evita duplicados y mantiene el orden)
if src_path not in sys.path:
    sys.path.append(src_path) 

from tufmax.core.engine import TUFMaxEngine
from tufmax.exceptions import TUFMaxGeometryError, TUFMaxError

def main():
    print("🚀 Iniciando TUFMax Hello World...")

    # 1. Coordenadas de la estación terrestre (Superficie)
    #ground_coords = (51.65, 141.8776, 2.250) # Lat, Lon, Alt(km)
    ground_coords = (51.65, 136.3976, 2.250) # Lat, Lon, Alt(km)

    
    # 2. Coordenadas SIMULADAS del satélite (Debe estar ARRIBA, no en el suelo)
    # Usamos la misma Lat/Lon pero una altura de 400 km (órbita baja típica)
    # NOTA: El motor actualmente espera una tupla para ground_coords, 
    # pero para satélites sin TLE, necesitamos engañar al sistema o usar un TLE real.
    
    # OPCIÓN A: Usar un TLE Real (Recomendado para evitar errores de lógica)
    # Descomenta esto y comenta la línea de tle_data=None
    tle_iss = [
        "1 25544U 98067A   26229.79251732  .00005860  00000-0  11255-3 0  9993",
        "2 25544  51.6334 355.1923 0007535  57.5442 302.6274 15.49477092581308"
    ]
    
    # OPCIÓN B: Si quieres forzar sin TLE, debes pasar coordenadas cartesianas 
    # o modificar la lógica para aceptar 'keplerian_data' simulado. 
    # Pero lo más fácil para probar AHORA es usar el TLE de arriba.

    # Configuración corregida
    config = {
        'min_elevation': 15.0,
        'dry_run': False,
        # Asegúrate de que ground_pos tenga altitud en metros si el logger lo espera así
        'ground_pos': {'lat': ground_coords[0], 'lon': ground_coords[1], 'alt': ground_coords[2]}, 
        'sat_pos': {'lat': 0.0, 'lon': 0.0, 'alt': 400000.0} # Solo informativo para el log
    }
    
    engine = TUFMaxEngine(config=config)

    # ----------------------------------------------------------------------
    # GUÍA RÁPIDA DE FORMATEO DE TIEMPO (target_time)
    # ----------------------------------------------------------------------
    # El motor acepta formatos flexibles. Elige el que mejor se adapte:
    #
    # 1. None (Por defecto):
    #    Usa la hora exacta de ejecución del script.
    #    Ej: target_time=None
    #
    # 2. Delta de tiempo (Futuro):
    #    Para simular 'dentro de 1 hora' o 'dentro de 0.92 días', usa datetime:
    #    import datetime
    #    target_time = datetime.datetime.now() + datetime.timedelta(hours=1)
    #    target_time = datetime.datetime.now() + datetime.timedelta(days=0.92)
    #
    # 3. Fecha Específica (String ISO):
    #    Ideal para reproducir experimentos pasados o futuros exactos.
    #    Formato: "YYYY-MM-DD HH:MM:SS"
    #    Ej: target_time="2024-12-31 23:59:59"
    #
    # 4. Objeto datetime:
    #    Si ya tienes un objeto datetime.python, pásalo directamente.
    # ----------------------------------------------------------------------

    try:
        result = engine.run(
            tle_data=tle_iss,             # <-- ¡USA ESTO! Es vital para tener un satélite real
            ground_coords=ground_coords,  
            target_time="2026-08-19 07:01:16",
            n_samples=50
        )


        # --- Validación del Resultado ---
        print("\n✅ ¡Ejecución Completada con Éxito!")
        print("-" * 40)
        print(result.summary())
        
        # Verificar datos geométricos
        # Nota: TUFMaxResult actual no tiene 'los_profile' directo a menos que lo agreguemos explícitamente
        # Pero podemos acceder a los datos a través de los atributos estándar si se llenaron correctamente
        if hasattr(result, 'altitude') and result.altitude is not None:
            print(f"\n📡 Datos del Perfil:")
            print(f"   - Puntos totales: {result.n_points}")
            
            # DEBUG: Inspeccionar el tipo y contenido antes de convertir
            alt_array = result.altitude
            
            # Asegurar que es un Quantity de Astropy. Si no lo es, intentar arreglarlo.
            if not hasattr(alt_array, 'to'):
                # Si es un array de numpy puro, asumimos que está en KM (por como lo devuelve los_path)
                # o en Metros. Dado tu log, parece que los_path devuelve KM.
                print(f"   [DEBUG] Altitude no es Quantity. Tipo: {type(alt_array)}")
                # Forzar conversión a Quantity si es necesario (asumiendo KM por defecto según tu código actual)
                alt_array = alt_array * u.km 
            
            # Extraer valores seguros
            try:
                # Convertir todo el array a km de una vez para evitar errores elemento a elemento
                alt_km_values = alt_array.to(u.km).value
                
                # Verificar si son números reales o basura
                if np.isnan(alt_km_values[-1]) or np.isinf(alt_km_values[-1]):
                    print(f"   ⚠️ ADVERTENCIA: El último valor de altitud es inválido (NaN/Inf)")
                    # Buscar el último valor válido
                    valid_mask = np.isfinite(alt_km_values)
                    if np.any(valid_mask):
                        last_valid_idx = np.where(valid_mask)[0][-1]
                        alt_start = alt_km_values[0] # O el primer válido
                        alt_end = alt_km_values[last_valid_idx]
                        print(f"   - Rango de altitud (ajustado): {alt_start:.2f} km a {alt_end:.2f} km")
                    else:
                        print(f"   - Rango de altitud: Todos los puntos son inválidos.")
                else:
                    alt_start = alt_km_values[0]
                    alt_end = alt_km_values[-1]
                    print(f"   - Rango de altitud: {alt_start:.2f} km a {alt_end:.2f} km")
                    
            except Exception as e:
                print(f"   ❌ ERROR al procesar altitudes: {e}")
                print(f"   Contenido crudo (primeros 3 elementos): {alt_array[:3]}")

        # --- DEBUG: Verificación Geométrica del Path ---
        if hasattr(result, 'los_profile') and result.los_profile is not None:
            profile = result.los_profile
            
            # 1. Obtener vectores de inicio y fin del path (en metros)
            r_start = profile.xyz[0]  # Debería ser muy cercano a Ground
            r_end = profile.xyz[-1]   # Debería ser muy cercano a Satélite
            
            # 2. Calcular longitud real del path (Magnitud del vector diferencia)
            # Fórmula: |r_end - r_start|
            delta_vec = r_end - r_start
            path_length_m = np.linalg.norm(delta_vec)
            path_length_km = path_length_m.to(u.km).value
            
            # 3. Obtener elevación calculada
            # Necesitamos recalcularla o extraerla si la guardaste, 
            # pero podemos inferirla de las altitudes si asumimos geometría simple,
            # o mejor, imprimimos los datos crudos.
            
            print(f"\n📏 VERIFICACIÓN GEOMÉTRICA DEL PATH:")
            print(f"   - Punto Inicio (km): {r_start.to(u.km).value}")
            print(f"   - Punto Fin (km):    {r_end.to(u.km).value}")
            print(f"   - Longitud Real del Path (Cuerda): {path_length_km:.2f} km")
            
            # 4. Calcular distancia teórica si fuera vertical (solo diferencia de alturas)
            h_start = profile.geodetic['alt'][0].to(u.km).value
            h_end = profile.geodetic['alt'][-1].to(u.km).value
            diff_altitude = abs(h_end - h_start)
            
            print(f"   - Diferencia de Altitud (Vertical): {diff_altitude:.2f} km")
            
            if path_length_km > diff_altitude * 1.05: # Si el path es >5% más largo que la vertical
                print(f"   ✅ El path es INCLINADO (Longitud > Diferencia de Altura)")
                ratio = path_length_km / diff_altitude
                print(f"      Factor de inclinación: {ratio:.2f}x (1.0 sería vertical puro)")
            else:
                print(f"   ⚠️  El path parece casi VERTICAL o hay un error geométrico.")
        
        else:
                print(f"\n❌ Nunca entró al if de los_profile!!!")


        # Verificar logs
        logs = [f for f in os.listdir('.') if f.startswith('TUFMax') and f.endswith('.log')]
        if logs:
            print(f"\n📄 Log de ejecución disponible en: {os.path.abspath(logs[0])}")

    except Exception as e:
        print(f"\n❌ Error durante la ejecución:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import numpy as np
    main()
