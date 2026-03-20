import os
import json
import pandas as pd
import argparse
from collections import Counter

def load_stats(model_name):
    """
    Carga las estadísticas del modelo desde el archivo JSON correspondiente.
    """
    stats_path = f"./stats/{model_name}_stats.json"
    with open(stats_path, 'r') as f:
        return json.load(f)

def extract_gantry_distance(path):
    """
    Extrae la distancia gantry desde el path del archivo.
    """
    return float(path.split('/')[2])  # Extrae la segunda posición del path


def calculate_tp_majority(row):
    """
    Calcula si la predicción por mayoría simple es un TP.
    """
    query_label = row["Original Label"]
    neighbor_labels = eval(row["Neighbor Labels"])
    predicted_label = Counter(neighbor_labels).most_common(1)[0][0]
    return query_label == predicted_label, predicted_label

# Función corregida para evitar división por cero
def safe_inverse(d, epsilon=1e-6):
    """
    Retorna la inversa de la distancia, evitando división por cero.
    Si d es 0, usa un pequeño epsilon.
    """
    return 1 / (d + epsilon)

def calculate_tp_distance(row, weight_func):
    """
    Calcula si la predicción ponderada por distancias es un TP.
    """
    query_label = row["Original Label"]
    neighbor_labels = eval(row["Neighbor Labels"])
    distances = eval(row["Distances"])

    weights = [weight_func(d) if d > 0 else 1.0 for d in distances]
    
    weighted_scores = {}
    for label, weight in zip(neighbor_labels, weights):
        weighted_scores[label] = weighted_scores.get(label, 0) + weight
    
    # Predicción con mayor peso
    predicted_label = max(weighted_scores, key=weighted_scores.get)
    return query_label == predicted_label, predicted_label


def calculate_tp_weighted_with_gantry(row, alpha=0.5, beta=0.5):
    """
    Calcula si la predicción ponderada por distancias combinadas (latente + gantry) es un TP.
    """
    query_label = row["Original Label"]
    query_gantry = extract_gantry_distance(row["Query"])
    neighbor_labels = eval(row["Neighbor Labels"])
    neighbor_distances = eval(row["Distances"])
    neighbor_paths = eval(row["Neighbors"])

    # Extraer distancias gantry para los vecinos
    neighbor_gantries = [extract_gantry_distance(path) for path in neighbor_paths]

    # Calcular pesos combinados evitando división por cero
    combined_weights = [
        calculate_combined_weight(
            latent_distance if latent_distance > 0 else 1e-6,  
            abs(query_gantry - neighbor_gantry) if abs(query_gantry - neighbor_gantry) > 0 else 1e-6,  
        )
        for latent_distance, neighbor_gantry in zip(neighbor_distances, neighbor_gantries)
    ]

    # Sumar puntuaciones por clase
    weighted_scores = {}
    for label, weight in zip(neighbor_labels, combined_weights):
        weighted_scores[label] = weighted_scores.get(label, 0) + weight

    # Predicción con mayor puntuación
    predicted_label = max(weighted_scores, key=weighted_scores.get)

    return query_label == predicted_label, predicted_label


def calculate_combined_weight(latent_distance, gantry_difference, alpha=0.5, beta=0.5):
    """
    Calcula un peso combinado basado en la distancia latente y la diferencia gantry.
    """
    # Inversa de la distancia latente
    w_latent = 1 / (1 + latent_distance)
    # Inversa de la diferencia gantry
    w_gantry = 1 / (1 + (gantry_difference))
    # Peso combinado
    return alpha * w_latent + beta * w_gantry


def calculate_metrics(data, stats, weight_func):
    """
    Calcula métricas globales y por clase, incluyendo Recall, F1-Score y Weighted F1-Score.
    """
    metrics = {}
    class_metrics = {}
    class_distribution = stats["class_distribution"]

    data["TP_Majority"], data["Predicted_Majority"] = zip(*data.apply(calculate_tp_majority, axis=1))
    data["TP_distance"], data["Predicted_distance"] = zip(*data.apply(calculate_tp_distance, axis=1, weight_func=weight_func))

    # Métricas globales
    metrics["knn_accuracy_majority"] = data["TP_Majority"].mean()
    metrics["knn_accuracy_distance"] = data["TP_distance"].mean()

    weighted_f1_majority = []
    weighted_f1_distance = []

    total_precision_majority = 0
    total_recall_majority = 0
    total_precision_distance = 0
    total_recall_distance = 0
    total_classes = len(class_distribution)

    for label in class_distribution.keys():
        label_data = data[data["Original Label"] == int(label)]

        total_queries = len(label_data)
        total_in_class = class_distribution[label] if label in class_distribution else 0  

        if total_queries == 0 or total_in_class == 0:
            class_metrics[label] = {
                "precision_majority": 0,
                "recall_majority": 0,
                "f1_majority": 0,
                "precision_distance": 0,
                "recall_distance": 0,
                "f1_distance": 0,
                "total_queries": total_queries
            }
            continue  

        # Majority Voting
        tp_majority = label_data["TP_Majority"].sum()
        precision_majority = tp_majority / total_queries if total_queries > 0 else 0
        recall_majority = min(tp_majority / total_in_class, 1)  
        f1_majority = (2 * precision_majority * recall_majority) / max((precision_majority + recall_majority), 1e-6)

        # Weighted KNN
        tp_distance = label_data["TP_distance"].sum()
        precision_distance = tp_distance / total_queries if total_queries > 0 else 0
        recall_distance = min(tp_distance / total_in_class, 1)  
        f1_distance = (2 * precision_distance * recall_distance) / max((precision_distance + recall_distance), 1e-6)

        # Guardar métricas por clase
        class_metrics[label] = {
            "precision_majority": precision_majority,
            "recall_majority": recall_majority,
            "f1_majority": f1_majority,
            "precision_distance": precision_distance,
            "recall_distance": recall_distance,
            "f1_distance": f1_distance,
            "total_queries": total_queries
        }

        # F1 ponderado por clase
        weighted_f1_majority.append(f1_majority * total_in_class / sum(class_distribution.values()))
        weighted_f1_distance.append(f1_distance * total_in_class / sum(class_distribution.values()))

        # Sumar para promedios globales
        total_precision_majority += precision_majority
        total_recall_majority += recall_majority
        total_precision_distance += precision_distance
        total_recall_distance += recall_distance

    # Promedios globales
    metrics["knn_weighted_f1_majority"] = sum(weighted_f1_majority)  # Ponderado por tamaño de clase
    metrics["knn_weighted_f1_distance"] = sum(weighted_f1_distance)  # Ponderado por tamaño de clase

    metrics["knn_average_f1_majority"] = sum(weighted_f1_majority) / max(total_classes, 1)  
    metrics["knn_average_f1_distance"] = sum(weighted_f1_distance) / max(total_classes, 1)  

    metrics["knn_average_precision_majority"] = total_precision_majority / max(total_classes, 1)
    metrics["knn_average_recall_majority"] = total_recall_majority / max(total_classes, 1)
    metrics["knn_average_precision_distance"] = total_precision_distance / max(total_classes, 1)
    metrics["knn_average_recall_distance"] = total_recall_distance / max(total_classes, 1)

    metrics.update({"class_metrics": class_metrics})

    return metrics, data


def calculate_metrics_with_gantry(data, stats, alpha=0.5, beta=0.5):
    """
    Calcula métricas globales y por clase considerando la combinación de distancia latente y gantry.
    """
    metrics = {}
    class_metrics = {}
    class_distribution = stats["class_distribution"]

    # Calcular True Positives y predicciones con gantry
    data["TP_distance_gantry"], data["Predicted_Weighted_Gantry"] = zip(*data.apply(
        lambda row: calculate_tp_weighted_with_gantry(row, alpha=alpha, beta=beta), axis=1))

    # Métricas globales
    metrics["knn_accuracy_gantry"] = data["TP_distance_gantry"].mean()

    weighted_f1_gantry = []
    total_precision_gantry = 0
    total_recall_gantry = 0
    total_classes = len(class_distribution)

    for label in class_distribution.keys():
        label_data = data[data["Original Label"] == int(label)]
        total_queries = len(label_data)
        total_in_class = class_distribution[label] if label in class_distribution else 0 
        if total_queries == 0 or total_in_class == 0:
            class_metrics[label] = {
                "precision_gantry": 0,
                "recall_gantry": 0,
                "f1_gantry": 0,
                "total_queries": total_queries
            }
            continue  

        # Gantry Voting
        tp_gantry = label_data["TP_distance_gantry"].sum()
        precision_gantry = tp_gantry / total_queries if total_queries > 0 else 0
        recall_gantry = min(tp_gantry / total_in_class, 1) 
        f1_gantry = (2 * precision_gantry * recall_gantry) / max((precision_gantry + recall_gantry), 1e-6)

        class_metrics[label] = {
            "precision_gantry": precision_gantry,
            "recall_gantry": recall_gantry,
            "f1_gantry": f1_gantry,
            "total_queries": total_queries
        }

        # F1 ponderado por clase
        weighted_f1_gantry.append(f1_gantry * total_in_class / sum(class_distribution.values()))

        # Sumar para promedios globales
        total_precision_gantry += precision_gantry
        total_recall_gantry += recall_gantry

    # Promedios globales
    metrics["knn_weighted_f1_gantry"] = sum(weighted_f1_gantry)  # F1 ponderado por clase
    metrics["knn_average_precision_gantry"] = total_precision_gantry / max(total_classes, 1)  
    metrics["knn_average_recall_gantry"] = total_recall_gantry / max(total_classes, 1)  

    metrics.update({"class_metrics": class_metrics})

    return metrics, data


def save_misses(data, model_name, k, weight_type):
    """
    Guarda las instancias que no son aciertos en un archivo separado.
    """
    # Construir la condición dinámicamente según las columnas disponibles
    conditions = []
    
    if weight_type == "majority" and "TP_Majority" in data.columns:
        conditions.append(~data["TP_Majority"])
        columns_to_save = ["Query", "Original Label", "Predicted_Majority", "Neighbor Labels", "Distances", "Neighbors"]
    elif weight_type == "distance" and "TP_distance" in data.columns:
        conditions.append(~data["TP_distance"])
        columns_to_save = ["Query", "Original Label", "Predicted_distance", "Neighbor Labels", "Distances", "Neighbors"]
    elif weight_type == "gantry" and "TP_distance_gantry" in data.columns:
        conditions.append(~data["TP_distance_gantry"])
        columns_to_save = ["Query", "Original Label", "Predicted_Weighted_Gantry", "Neighbor Labels", "Distances", "Neighbors"]
    else:
        print(f"No se encontraron columnas relevantes para {weight_type}.")
        return

    # Combinar condiciones si es necesario
    if conditions:
        combined_condition = conditions[0]
    else:
        print("No se pudieron construir condiciones para filtrar misses.")
        return

    # Aplicar el filtro
    misses = data[combined_condition]

    # Guardar las filas de fallos en un archivo CSV
    misses_path = f"./results/misses_{model_name}_{weight_type}.csv"
    misses[columns_to_save].to_csv(misses_path, index=False)
    print(f"Instancias fallidas guardadas en {misses_path}")


    
def save_matches(data, model_name, k, weight_type):
    """
    Guarda las instancias que son aciertos en un archivo separado.
    """
    # Construir la condición dinámicamente según las columnas disponibles
    conditions = []
    
    if weight_type == "majority" and "TP_Majority" in data.columns:
        conditions.append(data["TP_Majority"])
        columns_to_save = ["Query", "Original Label", "Predicted_Majority", "Neighbor Labels", "Distances", "Neighbors"]
    elif weight_type == "distance" and "TP_distance" in data.columns:
        conditions.append(data["TP_distance"])
        columns_to_save = ["Query", "Original Label", "Predicted_distance", "Neighbor Labels", "Distances", "Neighbors"]
    elif weight_type == "gantry" and "TP_distance_gantry" in data.columns:
        conditions.append(data["TP_distance_gantry"])
        columns_to_save = ["Query", "Original Label", "Predicted_Weighted_Gantry", "Neighbor Labels", "Distances", "Neighbors"]
    else:
        print(f"No se encontraron columnas relevantes para {weight_type}.")
        return

    # Combinar condiciones si es necesario
    if conditions:
        combined_condition = conditions[0]
    else:
        print("No se pudieron construir condiciones para filtrar matches.")
        return

    # Aplicar el filtro
    matches = data[combined_condition]

    # Guardar las filas de aciertos en un archivo CSV
    matches_path = f"./results/matches_{model_name}_{weight_type}.csv"
    matches[columns_to_save].to_csv(matches_path, index=False)
    print(f"Instancias correctas guardadas en {matches_path}")


def main():
    # Parsear argumentos
    parser = argparse.ArgumentParser(description="Cálculo de métricas de recuperación de información.")
    parser.add_argument("model_name", type=str, help="Nombre del modelo (e.g., pointnet_20250124-170053.pth)")
    args = parser.parse_args()

    # Cargar estadísticas del modelo
    model_name = args.model_name
    stats = load_stats(model_name)

    # Cargar archivo CSV
    csv_path = f"./results/k_neighbors_all_{model_name}.csv"
    if not os.path.exists(csv_path):
        print(f"Archivo CSV no encontrado: {csv_path}")
        return

    data = pd.read_csv(csv_path)

    best_k_inverse = None
    best_metric_inverse = 0
    best_k_squared = None
    best_metric_squared = 0
    results = {"inverse": {}, "squared": {}, "gantry": {}}
    best_k_gantry = None
    best_metric_gantry = 0

    for k in range(2, 21):
        print(f"Calculando métricas para k={k}...")

        # Crear una copia de los datos para el k actual
        data_k = data.copy()
        data_k["Neighbor Labels"] = data_k["Neighbor Labels"].apply(lambda x: str(eval(x)[:k]))
        data_k["Distances"] = data_k["Distances"].apply(lambda x: str(eval(x)[:k]))

        # Métricas con inversa de la distancia
        metrics_inverse, data_with_tps_inverse = calculate_metrics(
            data_k,
            stats,
            weight_func=lambda d: 1 / (d + 1e-6)  

        )
        results["inverse"][k] = metrics_inverse
        if metrics_inverse["knn_weighted_f1_distance"] > best_metric_inverse:
            best_metric_inverse = metrics_inverse["knn_weighted_f1_distance"]
            best_k_inverse = k
            save_misses(data_with_tps_inverse, model_name, k, "inverse")
            save_matches(data_with_tps_inverse, model_name, k, "inverse")

        print(f"Métricas para k={k} (Inverse): {metrics_inverse['knn_weighted_f1_distance']}")

        metrics_squared, data_with_tps_squared = calculate_metrics(
            data_k,
            stats,
            weight_func=lambda d: 1 / ((d ** 2) + 1e-6)  
        )
        results["squared"][k] = metrics_squared

        if metrics_squared["knn_weighted_f1_distance"] > best_metric_squared:
            best_metric_squared = metrics_squared["knn_weighted_f1_distance"]
            best_k_squared = k
            save_misses(data_with_tps_squared, model_name, k, "squared")
            save_matches(data_with_tps_squared, model_name, k, "squared")

        print(f"Métricas para k={k} (Squared): {metrics_squared['knn_weighted_f1_distance']}")

        # Métricas con gantry
        metrics_gantry, data_with_tps_gantry = calculate_metrics_with_gantry(
            data_k,
            stats,
            alpha=0.7,
            beta=0.3
        )
        results["gantry"][k] = metrics_gantry

        if metrics_gantry["knn_weighted_f1_gantry"] > best_metric_gantry:
            best_metric_gantry = metrics_gantry["knn_weighted_f1_gantry"]
            best_k_gantry = k
            save_misses(data_with_tps_gantry, model_name, k, "gantry")
            save_matches(data_with_tps_gantry, model_name, k, "gantry")

        print(f"Métricas para k={k} (Gantry): {metrics_gantry['knn_weighted_f1_gantry']}")

    # Guardar métricas finales con las mejores k al inicio
    final_metrics = {
        "best_k_inverse": best_k_inverse,
        "best_k_squared": best_k_squared,
        "best_k_gantry": best_k_gantry,
        "metrics_inverse": results["inverse"],
        "metrics_squared": results["squared"],
        "metrics_gantry": results["gantry"]
    }

    output_metrics_path = f"./results/metrics_{model_name}.json"
    with open(output_metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=4)
    print(f"Métricas guardadas en {output_metrics_path}")

if __name__ == "__main__":
    main()
