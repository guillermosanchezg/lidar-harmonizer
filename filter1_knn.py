import csv
import os
import sys

def load_csv(file_path):
    """Carga un archivo CSV en memoria."""
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
    return data

def save_csv(file_path, data, fieldnames):
    """Guarda los datos en un archivo CSV."""
    with open(file_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def extract_id_prefix(query_path):
    """Extrae el prefijo de ID base de un archivo Query."""
    base_name = os.path.basename(query_path)
    return base_name.split("_")[0]  # Toma solo la parte antes del primer "_"

def extract_date_and_hour(query_path):
    """Extrae el día y la hora de la ruta Query."""
    parts = query_path.split("/")
    return parts[-2], parts[-3]  

def filtro1(modelname):
    """Aplica el filtro utilizando matches en vez del dataset completo."""
    # Rutas hardcodeadas
    input_csv = f"./results/misses_{modelname}_gantry.csv"
    matches_csv = f"./results/matches_{modelname}_gantry.csv"
    output_csv = f"./results/misses_{modelname}_gantry_filtro1.csv"

    # Cargar archivos CSV
    data = load_csv(input_csv)
    matches_data = load_csv(matches_csv)

    # Separar instancias con -3.68 en el Query
    instances_at_minus_3_68 = [row for row in data if "-3.68" in row['Query']]

    # Filtrar las demás instancias
    filtered_data = [row for row in data if "-3.68" not in row['Query']]

    # Crear un diccionario estructurado con los matches
    matches_dict = {}
    for row in matches_data:
        path = row["Query"]
        label = row["Original Label"]
        id_prefix = extract_id_prefix(path)
        hour, date = extract_date_and_hour(path)
        
        key = (id_prefix, date, hour)
        if key not in matches_dict:
            matches_dict[key] = []

        matches_dict[key].append((path, label))

    # Filtrar las instancias según la coincidencia de etiquetas con matches
    non_matching_instances = []
    i = 0
    for row in filtered_data:
        query_path = row['Query']
        original_label = row['Original Label']
        id_prefix = extract_id_prefix(query_path)
        hour, date = extract_date_and_hour(query_path)    

        # Buscar en matches las instancias del mismo día y hora con el mismo ID base pero con diferente gantry_distance
        key = (id_prefix, date, hour)
        if key in matches_dict:
            print(f"Procesando Query: {query_path}")
            print(f"Key encontrado en matches: {key}")

            # Extraer el gantry_distance actual de la Query
            query_gantry_distance = query_path.split("/")[-5]  # El nivel de gantry_distance en la ruta
            print(f"Gantry Distance de la Query actual: {query_gantry_distance}")

            matching_labels = []
            for path, label in matches_dict[key]:
                gantry_distance = path.split("/")[-5]  # Extraer el gantry_distance del match
                if gantry_distance != query_gantry_distance:  # Solo considerar otras distancias
                    print(f"Revisando instancia en matches -> Path: {path}, Label: {label}, Gantry Distance: {gantry_distance}")
                    matching_labels.append(label)

            print(f"Matching Labels acumulados: {matching_labels}")
            print(f"Original Label de la instancia: {original_label}")

            # Comparar etiquetas
            if original_label not in matching_labels:
                print("Etiqueta no coincide. Agregando al resultado.")
                non_matching_instances.append(row)
            else:
                i += 1
                print(f"Etiqueta coincide. Instancia filtrada. Total filtradas: {i}")
        else:
            # Traza cuando el key no está en los matches
            print(f"Key no encontrado en matches: {key}")

    # Combinar los resultados
    final_data = instances_at_minus_3_68 + non_matching_instances
    print(non_matching_instances)

    # Guardar el CSV final
    save_csv(output_csv, final_data, fieldnames=data[0].keys())

    print(f"Archivo filtrado guardado en {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python filtro1.py <modelname>")
        sys.exit(1)

    modelname = sys.argv[1]
    print(f"Ejecutando filtro1 para {modelname}...")
    filtro1(modelname)
    print("Filtro aplicado exitosamente.")
