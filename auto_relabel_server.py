import os
import shutil
import pandas as pd
import argparse

ORDEN_DISTANCIAS = ["-13.68", "-8.68", "-3.68", "3.68", "8.68", "13.68"]

def extraer_id_vehiculo(path_pcd):
    partes = os.path.normpath(path_pcd).split(os.sep)
    if len(partes) >= 3:
        return os.sep.join(partes[-3:])
    return None

def buscar_y_mover_relacionados(id_vehiculo, nueva_clase_fine, dataset_base):
    archivos_movidos = 0
    for distancia in ORDEN_DISTANCIAS:
        dir_distancia = os.path.join(dataset_base, distancia)
        if not os.path.exists(dir_distancia):
            continue
            
        for clase_actual in os.listdir(dir_distancia):
            ruta_origen = os.path.join(dir_distancia, clase_actual, id_vehiculo)
            
            if os.path.exists(ruta_origen):
                clase_str = str(nueva_clase_fine)
                
                # Si ya está en la clase correcta, lo contamos pero no lo movemos
                if clase_actual == clase_str:
                    archivos_movidos += 1
                    continue
                    
                ruta_destino = os.path.join(dir_distancia, clase_str, id_vehiculo)
                os.makedirs(os.path.dirname(ruta_destino), exist_ok=True)
                try:
                    shutil.move(ruta_origen, ruta_destino)
                    archivos_movidos += 1
                except Exception as e:
                    print(f"  [ERROR] Fallo al mover a {distancia}: {e}")

    return archivos_movidos

def main():
    parser = argparse.ArgumentParser(description="Reetiquetador Automático en Servidor")
    parser.add_argument("csv_path", type=str, help="Ruta al relabel_candidates CSV")
    parser.add_argument("--dataset", type=str, default="../dataset_labeled", help="Ruta al dataset")
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"❌ Error: No se encontró el CSV en {args.csv_path}")
        return

    print("Cargando lista de candidatos...")
    df = pd.read_csv(args.csv_path)

    # 1. Extraer el ID único del vehículo para no procesar el mismo coche 6 veces
    df['id_vehiculo'] = df['Path'].apply(extraer_id_vehiculo)

    # 2. Agrupar por vehículo y quedarnos con la fila de mayor confianza
    # (Por si la IA dudó un poco más a 13 metros que a 3 metros)
    idx_max_conf = df.groupby('id_vehiculo')['Model_Confidence'].idxmax()
    df_unique = df.loc[idx_max_conf].copy()

    total_vehiculos = len(df_unique)
    print(f"Se evaluarán {total_vehiculos} vehículos únicos.")

    vehiculos_a_error = 0
    vehiculos_a_13 = 0
    archivos_totales_movidos = 0
    lista_manual = []

    for _, row in df_unique.iterrows():
        id_veh = row['id_vehiculo']
        conf = row['Model_Confidence']
        old_name = row['Old_Label_Name']
        pred_name = row['Predicted_Label_Name']

        # --- REGLA 1: Baja confianza (< 60%) -> Descartar a clase 0 ---
        if conf < 0.60:
            movidos = buscar_y_mover_relacionados(id_veh, 0, args.dataset)
            vehiculos_a_error += 1
            archivos_totales_movidos += movidos

        # --- REGLA 2: Alta confianza (>= 60%) ---
        else:
            # Transiciones directas y seguras a L3H3 (Van -> 13)
            if pred_name == 'Van' and old_name in ['Passenger Car', 'Bus']:
                movidos = buscar_y_mover_relacionados(id_veh, 13, args.dataset)
                vehiculos_a_13 += 1
                archivos_totales_movidos += movidos
            
            # --- REGLA 3: Transiciones ambiguas -> Revisión manual ---
            else:
                # Quitamos la columna extra que añadimos antes de guardar
                row_dict = row.drop('id_vehiculo').to_dict()
                lista_manual.append(row_dict)

    # --- RESUMEN Y EXPORTACIÓN ---
    print("\n=======================================================")
    print("  RESULTADOS DEL AUTO-REETIQUETADO (SERVIDOR)")
    print("=======================================================")
    print(f" 🗑️ Vehículos movidos a ERROR (0) por baja confianza: {vehiculos_a_error}")
    print(f" 🚐 Vehículos auto-corregidos a L3H3 (13): {vehiculos_a_13}")
    print(f" 📁 Total de archivos .pcd reubicados en disco: {archivos_totales_movidos}")
    print(f" ⚠️ Vehículos ambiguos reservados para revisión: {len(lista_manual)}")
    print("=======================================================")

    if lista_manual:
        out_csv = args.csv_path.replace(".csv", "_manual_review.csv")
        pd.DataFrame(lista_manual).to_csv(out_csv, index=False)
        print(f"\n✅ Archivo generado para descargar a local: {out_csv}")

if __name__ == "__main__":
    main()
