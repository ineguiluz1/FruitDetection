import cv2
import os
import numpy as np
from pathlib import Path

# Configuración de rutas
base_dir = os.path.dirname(os.path.abspath(__file__))
raw_dir = os.path.join(base_dir, '..', 'data', 'real', 'raw')
processed_dir = os.path.join(base_dir, '..', 'data', 'real', 'processed')

# Crear directorio de salida si no existe
os.makedirs(processed_dir, exist_ok=True)

def preprocess_image(img_path, output_path, target_size=(100, 100)):
    """
    Preprocesa una imagen: redimensiona a 100x100 y normaliza píxeles.
    
    Args:
        img_path: str - Ruta a la imagen de entrada
        output_path: str - Ruta donde guardar la imagen procesada
        target_size: tuple - Tamaño objetivo (ancho, alto)
    """
    # Cargar imagen
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {img_path}")
    
    # Convertir de BGR a RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Redimensionar a 100x100
    img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Normalizar píxeles a rango [0, 1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Convertir de vuelta a uint8 para guardar (multiplicar por 255)
    img_to_save = (img_normalized * 255).astype(np.uint8)
    
    # Convertir de RGB a BGR para cv2.imwrite
    img_bgr = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
    
    # Guardar imagen procesada
    cv2.imwrite(output_path, img_bgr)
    print(f"Procesada: {os.path.basename(img_path)} -> {os.path.basename(output_path)}")

if __name__ == "__main__":
    # Obtener todas las imágenes en el directorio raw
    image_files = [f for f in os.listdir(raw_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"No se encontraron imágenes en {raw_dir}")
    else:
        print(f"Procesando {len(image_files)} imágenes...")
        
        for img_name in image_files:
            img_path = os.path.join(raw_dir, img_name)
            output_path = os.path.join(processed_dir, img_name)
            
            try:
                preprocess_image(img_path, output_path)
            except Exception as e:
                print(f"Error procesando {img_name}: {str(e)}")
        
        print(f"\n¡Procesamiento completado! Imágenes guardadas en {processed_dir}")
