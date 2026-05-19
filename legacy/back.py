import os
import json
import pandas as pd
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

def calculate_tp_distance(row, weight_func):
    """
    Calcula si la predicción ponderada por distancias es un TP.
    """
    query_label = row["Original Label"]
    neighbor_labels = eval(row["Neighbor Labels"])
    distances = eval(row["Distances"])

    # Calcular pesos usando la función de peso proporcionada
    weights = [weight_func(d) for d in distances]
    weighted_scores = {}
    for label, weight in zip(neighbor_labels, weights):
        weighted_scores[label] = weighted_scores.get(label, 0) + weight
    
    # Predicción con mayor peso
    predicted_label = max(weighted_scores, key=weighted_scores.get)
    return query_label == predicted_label, predicted_label

def dynamic_alpha_beta(class_size, total_size):
    """
    Ajusta dinámicamente los valores de alpha y beta según el tamaño de la clase.
    """
    alpha = class_size / total_size
    beta = 1 - alpha
    return alpha, beta

def calculate_tp_weighted_with_gantry(row, class_distribution, max_latent, max_gantry, alpha=0.5, beta=0.5):
    """
    Calcula si la predicción ponderada por distancias combinadas (latente + gantry) es un TP.
    """
    query_label = row["Original Label"]
    query_gantry = extract_gantry_distance(row["Query"])
    neighbor_labels = eval(row["Neighbor Labels"])
    neighbor_distances = eval(row["Distances"])
    neighbor_paths = eval(row["Neighbors"])

    # Convertir query_label al tipo correcto si es necesario
    query_label = str(query_label) if isinstance(list(class_distribution.keys())[0], str) else query_label

    # Extraer distancias gantry para los vecinos
    neighbor_gantries = [extract_gantry_distance(path) for path in neighbor_paths]

    # Número de instancias en la clase y total
    instances_in_class = class_distribution.get(query_label, 0)  # Usa 0 si la clase no existe
    total_instances = sum(class_distribution.values())

    # Ajustar alpha y beta dinámicamente
    alpha, beta = dynamic_alpha_beta(instances_in_class, total_instances)

    # Calcular pesos combinados
    combined_weights = [
        calculate_combined_weight(
            latent_distance=latent_distance,
            gantry_difference=abs(query_gantry - neighbor_gantry),
            max_latent=max_latent,
            max_gantry=max_gantry,
            alpha=alpha,
            beta=beta
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


def calculate_combined_weight(latent_distance, gantry_difference, max_latent, max_gantry, alpha=0.5, beta=0.5):
    normalized_latent = latent_distance / max_latent
    normalized_gantry = gantry_difference / max_gantry
    w_latent = 1 / (1 + normalized_latent)
    w_gantry = 1 / (1 + normalized_gantry)
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
    weighted_f1_distance= []

    total_precision_majority = 0
    total_recall_majority = 0
    total_precision_distance= 0
    total_recall_distance= 0
    total_classes = len(class_distribution)

    for label in class_distribution.keys():
        label_data = data[data["Original Label"] == int(label)]

        if label_data.empty:
            class_metrics[label] = {
                "precision_majority": 0,
                "recall_majority": 0,
                "f1_majority": 0,
                "precision_distance": 0,
                "recall_distance": 0,
                "f1_distance": 0,
                "total_queries": 0
            }
            continue

        total_queries = len(label_data)
        total_in_class = class_distribution[label]

        # Majority Voting
        tp_majority = label_data["TP_Majority"].sum()
        precision_majority = tp_majority / total_queries if total_queries > 0 else 0
        recall_majority = tp_majority / total_in_class if total_in_class > 0 else 0
        f1_majority = (2 * precision_majority * recall_majority) / (precision_majority + recall_majority) if (precision_majority + recall_majority) > 0 else 0

        # Weighted KNN
        tp_distance= label_data["TP_distance"].sum()
        precision_distance= tp_distance / total_queries if total_queries > 0 else 0
        recall_distance = tp_distance / total_in_class if total_in_class > 0 else 0
        f1_distance = (2 * precision_distance * recall_distance ) / (precision_distance + recall_distance ) if (precision_distance + recall_distance ) > 0 else 0

        # Guardar métricas por clase
        class_metrics[label] = {
            "precision_majority": precision_majority,
            "recall_majority": recall_majority,
            "f1_majority": f1_majority,
            "precision_distance": precision_distance ,
            "recall_distance": recall_distance ,
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

    metrics["knn_average_f1_majority"] = sum(f1_majority for f1_majority in weighted_f1_majority) / total_classes  # Promedio simple
    metrics["knn_average_f1_distance"] = sum(f1_distance for f1_distance in weighted_f1_distance) / total_classes  # Promedio simple

    metrics["knn_average_precision_majority"] = total_precision_majority / total_classes
    metrics["knn_average_recall_majority"] = total_recall_majority / total_classes
    metrics["knn_average_precision_distance"] = total_precision_distance / total_classes
    metrics["knn_average_recall_distance"] = total_recall_distance / total_classes


    metrics.update({
        "class_metrics": class_metrics
    })

    return metrics, data

def calculate_metrics_with_gantry(data, stats, max_latent, max_gantry, alpha=0.5, beta=0.5):
    """
    Calcula métricas globales y por clase considerando la combinación de latente y gantry.
    Incluye métricas ponderadas y medias simples.
    """
    metrics = {}
    class_metrics = {}
    class_distribution = stats["class_distribution"]

    data["TP_distance_gantry"], data["Predicted_Weighted_Gantry"] = zip(*data.apply(
        lambda row: calculate_tp_weighted_with_gantry(row, class_distribution, max_latent, max_gantry, alpha, beta), axis=1
    ))

    # Métricas globales
    metrics["knn_accuracy_gantry"] = data["TP_distance_gantry"].mean()

    weighted_f1_gantry = []
    average_f1_gantry = []
    total_precision_gantry = 0
    total_recall_gantry = 0
    total_classes = len(class_distribution)

    for label in class_distribution.keys():
        label_data = data[data["Original Label"] == int(label)]

        if label_data.empty:
            class_metrics[label] = {
                "precision_gantry": 0,
                "recall_gantry": 0,
                "f1_gantry": 0,
                "total_queries": 0
            }
            continue

        total_queries = len(label_data)
        total_in_class = class_distribution[label]

        tp_gantry = label_data["TP_distance_gantry"].sum()
        precision_gantry = tp_gantry / total_queries if total_queries > 0 else 0
        recall_gantry = tp_gantry / total_in_class if total_in_class > 0 else 0
        f1_gantry = (2 * precision_gantry * recall_gantry) / (precision_gantry + recall_gantry) if (precision_gantry + recall_gantry) > 0 else 0

        class_metrics[label] = {
            "precision_gantry": precision_gantry,
            "recall_gantry": recall_gantry,
            "f1_gantry": f1_gantry,
            "total_queries": total_queries
        }

        # F1 ponderado por clase
        weighted_f1_gantry.append(f1_gantry * total_in_class / sum(class_distribution.values()))

        # F1 promedio por clase (no ponderado)
        average_f1_gantry.append(f1_gantry)

        # Sumar para promedios globales
        total_precision_gantry += precision_gantry
        total_recall_gantry += recall_gantry

    # Métricas globales
    metrics["knn_weighted_f1_gantry"] = sum(weighted_f1_gantry)  # F1 ponderado por clase
    metrics["knn_average_f1_gantry"] = sum(average_f1_gantry) / total_classes  # F1 promedio simple
    metrics["knn_average_precision_gantry"] = total_precision_gantry / total_classes  # Precisión promedio
    metrics["knn_average_recall_gantry"] = total_recall_gantry / total_classes  # Recall promedio

    metrics.update({
        "class_metrics": class_metrics
    })

    return metrics, data


def save_misses(data, model_name, k, weight_type):
    """
    Guarda las instancias que no son aciertos en un archivo separado.
    """
    misses = data[(~data["TP_Majority"]) | (~data["TP_distance"])]
    misses_path = f"./results/misses_{model_name}_{weight_type}.csv"
    misses[["Query", "Original Label", "Predicted_Majority", "Predicted_distance", "Neighbor Labels", "Distances", "Neighbors"]].to_csv(misses_path, index=False)
    print(f"Instancias fallidas guardadas en {misses_path}")
    
def save_matches(data, model_name, k, weight_type):
    """
    Guarda las instancias que son aciertos en un archivo separado.
    """
    matches = data[(data["TP_Majority"]) & (data["TP_distance"])]
    matches_path = f"./results/matches_{model_name}_{weight_type}.csv"
    matches[["Query", "Original Label", "Predicted_Majority", "Predicted_distance", "Neighbor Labels", "Distances", "Neighbors"]].to_csv(matches_path, index=False)
    print(f"Instancias correctas guardadas en {matches_path}")

def main():
    # Parsear argumentos
    import argparse
    parser = argparse.ArgumentParser(description="Cálculo de métricas de recuperación de información.")
    parser.add_argument("model_name", type=str, help="Nombre del modelo (e.g., pointnet_20250124-170053.pth)")
    args = parser.parse_args()

    # Cargar estadísticas del modelo
    model_name = args.model_name
    stats = load_stats(model_name)

    # Cargar archivo CSV
    csv_path = f"./results/k_neighbors_{model_name}.csv"
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

    max_latent = max(data["Distances"].apply(lambda x: max(eval(x))).max(), 1e-6)
    max_gantry = max(data["Query"].apply(lambda x: abs(extract_gantry_distance(x))).max(), 1e-6)


    for k in range(2, 10):
        print(f"Calculando métricas para k={k}...")

        # Crear una copia de los datos para el k actual
        data_k = data.copy()
        data_k["Neighbor Labels"] = data_k["Neighbor Labels"].apply(lambda x: str(eval(x)[:k]))
        data_k["Distances"] = data_k["Distances"].apply(lambda x: str(eval(x)[:k]))

        # Métricas con inversa de la distancia
        metrics_inverse, data_with_tps_inverse = calculate_metrics(
            data_k,
            stats,
            weight_func=lambda d: 1 / d
        )
        results["inverse"][k] = metrics_inverse

        if metrics_inverse["knn_weighted_f1_distance"] > best_metric_inverse:
            best_metric_inverse = metrics_inverse["knn_weighted_f1_distance"]
            best_k_inverse = k
            save_misses(data_with_tps_inverse, model_name, k, "inverse")
            save_matches(data_with_tps_inverse, model_name, k, "inverse")

        print(f"Métricas para k={k} (Inverse): {metrics_inverse['knn_weighted_f1_distance']}")

        # Métricas con inversa del cuadrado de la distancia
        metrics_squared, data_with_tps_squared = calculate_metrics(
            data_k,
            stats,
            weight_func=lambda d: 1 / (d ** 2)
        )
        results["squared"][k] = metrics_squared

        if metrics_squared["knn_weighted_f1_distance"] > best_metric_squared:
            best_metric_squared = metrics_squared["knn_weighted_f1_distance"]
            best_k_squared = k
            save_misses(data_with_tps_squared, model_name, k, "squared")
            save_matches(data_with_tps_squared, model_name, k, "squared")

        print(f"Métricas para k={k} (Squared): {metrics_squared['knn_weighted_f1_distance']}")

        metrics_gantry, data_with_tps_gantry = calculate_metrics_with_gantry(
            data_k, stats, max_latent, max_gantry, alpha=0.5, beta=0.5
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
