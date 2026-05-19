import torch
import os

def inspect_models_in_directory(directory, output_file):
    """
    Itera sobre todos los archivos .pth en un directorio, extrae los parámetros
    de cada modelo y guarda los resultados en un archivo de texto.

    Args:
        directory (str): Ruta al directorio con los modelos .pth.
        output_file (str): Ruta al archivo de salida .txt.
    """
    try:
        with open(output_file, 'w') as f:
            for file_name in os.listdir(directory):
                if file_name.endswith('.pth'):
                    model_path = os.path.join(directory, file_name)
                    f.write(f"Parámetros del modelo: {file_name}\n")
                    try:
                        # Cargar los parámetros del modelo
                        model_params = torch.load(model_path, map_location=torch.device('cpu'))
                        
                        if isinstance(model_params, dict) and 'state_dict' in model_params:
                            model_params = model_params['state_dict']
                        
                        for param_name, param_value in model_params.items():
                            shape_info = param_value.shape if hasattr(param_value, 'shape') else 'Scalar'
                            f.write(f"  {param_name}: {shape_info}\n")
                    except Exception as e:
                        f.write(f"  Error al cargar el modelo: {e}\n")
                    f.write("\n")  # Línea en blanco entre modelos
        print(f"Parámetros escritos en {output_file}")
    except Exception as e:
        print(f"Error al procesar los modelos: {e}")

if __name__ == "__main__":
    # Directorio actual
    current_directory = os.getcwd()
    # Nombre del archivo de salida
    output_txt = os.path.join(current_directory, "model_parameters.txt")
    
    # Llamar a la función para procesar los modelos
    inspect_models_in_directory(current_directory, output_txt)
