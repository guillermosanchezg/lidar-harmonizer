import os
import yaml
import pandas as pd
from collections import Counter

def main():
    dataset_path = "../dataset_labeled"
    
    print("Cargando params.yaml...")
    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)
        
    class_name_mapping = params.get("CLASS_NAME_MAPPING", {})
    superclass_mapping = params.get("SUPERCLASS_MAPPING", {})
    superclass_names = params.get("SUPERCLASS_NAMES", {})
    
    print(f"Escaneando el directorio {dataset_path} (esto puede tardar unos segundos)...")
    raw_counts = Counter()
    
    # Escaneo rápido de archivos sin cargar las nubes en memoria
    for root, dirs, files in os.walk(dataset_path):
        pcd_count = sum(1 for f in files if f.endswith('.pcd'))
        if pcd_count > 0:
            parts = root.replace('\\', '/').split('/')
            try:
                # Asumimos estructura: dataset_labeled/<gantry>/<class_id>/...
                idx = parts.index(os.path.basename(dataset_path))
                class_id = parts[idx + 2]
                raw_counts[class_id] += pcd_count
            except (ValueError, IndexError):
                pass

    # Tabla Fine-Grained
    fg_rows = []
    for cid, count in raw_counts.items():
        name = class_name_mapping.get(int(cid), class_name_mapping.get(str(cid), f"Clase {cid}"))
        fg_rows.append({"ID Carpeta": cid, "Clase (Fine-Grained)": name, "Instancias": count})
        
    fg_df = pd.DataFrame(fg_rows).sort_values(by="Instancias", ascending=False)
    
    # Tabla Superclass
    sc_counts = Counter()
    for cid, count in raw_counts.items():
        sc_id = superclass_mapping.get(int(cid), superclass_mapping.get(str(cid), None))
        if sc_id is not None:
            sc_name = superclass_names.get(int(sc_id), superclass_names.get(str(sc_id), f"Superclase {sc_id}"))
            sc_counts[sc_name] += count
        else:
            sc_counts[f"Sin Superclase (ID {cid})"] += count
            
    sc_rows = [{"Superclase": name, "Instancias": count} for name, count in sc_counts.items()]
    sc_df = pd.DataFrame(sc_rows).sort_values(by="Instancias", ascending=False)
    
    print("\n=========================================================")
    print(" DISTRIBUCIÓN FINE-GRAINED (15 clases originales)")
    print("=========================================================")
    print(fg_df.to_string(index=False))
    print(f"\nTotal vehículos: {fg_df['Instancias'].sum()}")
    
    print("\n=========================================================")
    print(" DISTRIBUCIÓN SUPERCLASS (Clases agrupadas)")
    print("=========================================================")
    print(sc_df.to_string(index=False))
    print(f"\nTotal vehículos: {sc_df['Instancias'].sum()}")
    print("=========================================================\n")

if __name__ == '__main__':
    main()

