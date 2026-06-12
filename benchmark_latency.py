import time
import numpy as np
import pandas as pd
import os
from sklearn.neighbors import NearestNeighbors

def benchmark_system_latency(knowledge_base_size=5000, query_size=100, feature_dim=1024, max_k=10, output_csv="latency_benchmark_results.csv"):
    print(f"=== INICIANDO BENCHMARK DE LATENCIA ===")
    print(f"Tamaño Base de Conocimiento: {knowledge_base_size} nubes")
    print(f"Consultas a simular: {query_size} nubes\n")

    # 1. Simular Embeddings
    # PointNet/PointNet++ tarda ~5-15 ms por nube. Simulamos los embeddings para aislar el coste del KNN.
    kb_embeddings = np.random.rand(knowledge_base_size, feature_dim).astype(np.float32)
    query_embeddings = np.random.rand(query_size, feature_dim).astype(np.float32)

    # 2. Entrenar el buscador KNN (Tiempo de inicialización)
    start_fit = time.time()
    knn = NearestNeighbors(n_neighbors=max_k, metric='euclidean', algorithm='auto', n_jobs=-1)
    knn.fit(kb_embeddings)
    fit_time = (time.time() - start_fit) * 1000
    print(f"[Info] Indexar KB tardó: {fit_time:.2f} ms")

    # Preparar diccionario para guardar resultados
    results_list = []

    # 3. Medir tiempo de Búsqueda KNN según K
    print("\n--- Tiempos de Búsqueda KNN por nube ---")
    for k in range(2, max_k + 1):
        # Búsqueda en bloque
        start_search = time.time()
        distances, indices = knn.kneighbors(query_embeddings, n_neighbors=k)
        end_search = time.time()
        
        # Calcular media por nube
        time_per_query = ((end_search - start_search) * 1000) / query_size
        print(f"Para K={k:<2} -> Tiempo KNN: {time_per_query:.4f} ms / nube")

        # 4. Medir tiempo de la lógica de Votación (Simulación Regla Inversa/Gantry)
        start_rules = time.time()
        for d_row, i_row in zip(distances, indices):
            w_inv = [1 / (d + 1e-6) for d in d_row]
            scores = {}
            for idx, w in zip(i_row, w_inv):
                scores[idx] = scores.get(idx, 0) + w
            _ = max(scores, key=scores.get)
        end_rules = time.time()
        
        rules_time_per_query = ((end_rules - start_rules) * 1000) / query_size
        total_time_ms = time_per_query + rules_time_per_query

        # Guardar en la lista
        results_list.append({
            "KB_Size": knowledge_base_size,
            "K_Value": k,
            "Fit_Time_ms": round(fit_time, 4),
            "KNN_Time_ms": round(time_per_query, 4),
            "Rules_Time_ms": round(rules_time_per_query, 4),
            "Total_Inference_ms": round(total_time_ms, 4)
        })

    print(f"\n[Info] Aplicar reglas (media K=10) tarda aprox: {results_list[-1]['Rules_Time_ms']:.4f} ms / nube")
    
    # 5. Guardar/Añadir a CSV
    df_new = pd.DataFrame(results_list)
    
    # Si el archivo ya existe, lo añadimos sin sobreescribir la cabecera
    if os.path.exists(output_csv):
        df_new.to_csv(output_csv, mode='a', header=False, index=False)
        print(f"\n[+] Resultados AÑADIDOS a {output_csv}")
    else:
        df_new.to_csv(output_csv, mode='w', header=True, index=False)
        print(f"\n[+] Resultados CREADOS en {output_csv}")
        
    print("=======================================\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark KNN Latency')
    parser.add_argument('--kb_size', type=int, default=5000, help='Tamaño de la Base de Conocimiento')
    args = parser.parse_args()
    
    benchmark_system_latency(knowledge_base_size=args.kb_size)
