#!/usr/bin/env python3
"""
Script para procesar imagenes capturadas, extraer caracteristicas y predecir con Random Forest.

Pasos:
1. Cargar imagenes del directorio 'captures'
2. Aplicar pixel normalization
3. Escalar al tamaño estandar (100x100 como en Fruits 360)
4. Aplicar el mismo preprocesamiento
5. Extraer caracteristicas (geometricas, color, textura)
6. Guardar caracteristicas en CSV
7. Cargar modelo Random Forest del .pkl
8. Generar predicciones
"""

import cv2
import numpy as np
import os
import pandas as pd
from pathlib import Path
import joblib
from skimage.measure import shannon_entropy
from skimage.feature import graycomatrix, graycoprops


# ============================================================================
# CONFIGURACION DE RUTAS
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTURES_DIR = os.path.join(BASE_DIR, '..', 'captures')
OUTPUT_CSV = os.path.join(BASE_DIR, '..', 'captures', 'features.csv')
PREDICTIONS_CSV = os.path.join(BASE_DIR, '..', 'captures', 'predictions.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'random_forest_classifier.pkl')
MAPPING_PATH = os.path.join(BASE_DIR, 'models', 'label_mapping.pkl')

# Tamaño estandar de Fruits 360
TARGET_SIZE = (100, 100)


# ============================================================================
# FUNCIONES DE CARGA Y PREPROCESAMIENTO
# ============================================================================

def load_image(img_path):
    """
    Carga una imagen desde el path y la convierte a RGB.
    
    Args:
        img_path: Ruta a la imagen
        
    Returns:
        img: Imagen en formato RGB (0-255)
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def normalize_image(img):
    """
    Normaliza la imagen al rango [0, 1].
    
    Args:
        img: Imagen de entrada (0-255)
        
    Returns:
        normalized_img: Imagen normalizada (0-1)
    """
    normalized_img = img.astype(np.float32) / 255.0
    return normalized_img


def remove_background(img):
    """
    Elimina el fondo (asume fondo blanco como en las capturas).
    
    Args:
        img: Imagen RGB de entrada
        
    Returns:
        mask: Mascara binaria de la fruta
    """
    # Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Umbralizado inverso para separar fruta del fondo blanco
    # Fondo blanco = 255, fruta = valores mas bajos
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Limpiar ruido con morfologia
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return mask


def resize_image(img, target_size=TARGET_SIZE):
    """
    Redimensiona la imagen manteniendo aspect ratio y centrandola.
    
    Args:
        img: Imagen de entrada
        target_size: Tupla (width, height) del tamaño objetivo
        
    Returns:
        resized: Imagen redimensionada con fondo blanco
    """
    h, w = img.shape[:2]
    target_w, target_h = target_size
    
    # Calcular aspect ratio
    aspect = w / h
    target_aspect = target_w / target_h
    
    # Redimensionar manteniendo aspect ratio
    if aspect > target_aspect:
        # Imagen mas ancha
        new_w = target_w
        new_h = int(target_w / aspect)
    else:
        # Imagen mas alta
        new_h = target_h
        new_w = int(target_h * aspect)
    
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Crear imagen con fondo blanco
    final_img = np.ones((target_h, target_w, 3), dtype=img.dtype) * 255
    
    # Centrar la imagen redimensionada
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    
    final_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
    
    return final_img


# ============================================================================
# EXTRACCION DE CARACTERISTICAS
# ============================================================================

def extract_geometric_features(img, mask=None):
    """
    Extrae caracteristicas geometricas: circularidad y area.
    
    Args:
        img: Imagen RGB
        mask: Mascara binaria (opcional)
        
    Returns:
        features: Dict con circularidad y area
        contour: Contorno mas grande
        mask: Mascara utilizada
    """
    if mask is None:
        mask = remove_background(img)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {'circularity': 0.0, 'area': 0.0}, None, mask
    
    # Obtener el contorno mas grande
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calcular area y perimetro
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    
    # Calcular circularidad: 4*pi*area / perimetro^2
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
    
    return {
        'circularity': circularity,
        'area': area
    }, largest_contour, mask


def extract_color_features(img, mask=None, bins=8):
    """
    Extrae caracteristicas de color en espacio HSV.
    
    Args:
        img: Imagen RGB
        mask: Mascara binaria (opcional)
        bins: Numero de bins para histogramas
        
    Returns:
        color_features: Dict con caracteristicas de color
    """
    if mask is None:
        mask = remove_background(img)
    
    # Convertir a HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # Extraer canales solo de la fruta
    h_channel = hsv[:, :, 0][mask > 0]
    s_channel = hsv[:, :, 1][mask > 0]
    v_channel = hsv[:, :, 2][mask > 0]
    
    # Calcular estadisticas
    h_mean = np.mean(h_channel) if len(h_channel) > 0 else 0.0
    s_mean = np.mean(s_channel) if len(s_channel) > 0 else 0.0
    v_mean = np.mean(v_channel) if len(v_channel) > 0 else 0.0
    
    h_std = np.std(h_channel) if len(h_channel) > 0 else 0.0
    s_std = np.std(s_channel) if len(s_channel) > 0 else 0.0
    v_std = np.std(v_channel) if len(v_channel) > 0 else 0.0
    
    # Calcular histogramas
    h_hist, _ = np.histogram(h_channel, bins=bins, range=(0, 180))
    h_hist = h_hist.astype(float) / (np.sum(h_hist) + 1e-7)
    
    s_hist, _ = np.histogram(s_channel, bins=bins, range=(0, 256))
    s_hist = s_hist.astype(float) / (np.sum(s_hist) + 1e-7)
    
    v_hist, _ = np.histogram(v_channel, bins=bins, range=(0, 256))
    v_hist = v_hist.astype(float) / (np.sum(v_hist) + 1e-7)
    
    # Crear diccionario de caracteristicas
    color_features = {
        'h_mean': h_mean,
        's_mean': s_mean,
        'v_mean': v_mean,
        'h_std': h_std,
        's_std': s_std,
        'v_std': v_std,
    }
    
    # Agregar histogramas
    for i, val in enumerate(h_hist):
        color_features[f'h_hist_bin{i}'] = val
    for i, val in enumerate(s_hist):
        color_features[f's_hist_bin{i}'] = val
    for i, val in enumerate(v_hist):
        color_features[f'v_hist_bin{i}'] = val
    
    return color_features


def extract_texture_features(img, mask=None, distance=1):
    """
    Extrae caracteristicas de textura usando Haralick y entropia.
    
    Args:
        img: Imagen RGB
        mask: Mascara binaria (opcional)
        distance: Distancia para GLCM
        
    Returns:
        texture_features: Dict con caracteristicas de textura
    """
    if mask is None:
        mask = remove_background(img)
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Extraer solo pixeles de la fruta
    fruit_pixels = gray[mask > 0]
    
    # Calcular entropia
    entropy = shannon_entropy(fruit_pixels) if len(fruit_pixels) > 0 else 0.0
    
    # Recortar region de la fruta para GLCM
    coords = np.where(mask > 0)
    if len(coords[0]) > 0:
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        gray_crop = gray[y_min:y_max+1, x_min:x_max+1]
    else:
        gray_crop = gray
    
    # Calcular GLCM y propiedades de Haralick
    try:
        glcm = graycomatrix(
            gray_crop,
            distances=[distance],
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            levels=256,
            symmetric=True,
            normed=True
        )
        
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
    except:
        contrast = dissimilarity = homogeneity = energy = 0.0
    
    return {
        'haralick_contrast': contrast,
        'haralick_dissimilarity': dissimilarity,
        'haralick_homogeneity': homogeneity,
        'haralick_energy': energy,
        'entropy': entropy
    }


def extract_all_features(img, mask=None, bins=8, distance=1):
    """
    Extrae todas las caracteristicas (geometricas + color + textura).
    
    Args:
        img: Imagen RGB
        mask: Mascara binaria (opcional)
        bins: Numero de bins para histogramas
        distance: Distancia para GLCM
        
    Returns:
        all_features: Dict con todas las caracteristicas
        mask: Mascara utilizada
    """
    if mask is None:
        mask = remove_background(img)
    
    # Extraer caracteristicas geometricas
    geom_features, _, mask = extract_geometric_features(img, mask)
    
    # Extraer caracteristicas de color
    color_features = extract_color_features(img, mask, bins)
    
    # Extraer caracteristicas de textura
    texture_features = extract_texture_features(img, mask, distance)
    
    # Combinar todas las caracteristicas
    all_features = {**geom_features, **color_features, **texture_features}
    
    return all_features, mask


# ============================================================================
# PROCESAMIENTO DE DATASET E INFERENCIA
# ============================================================================

def process_captures(captures_dir, output_csv, bins=8, distance=1):
    """
    Procesa todas las imagenes en el directorio captures y guarda caracteristicas en CSV.
    
    Args:
        captures_dir: Directorio con imagenes capturadas
        output_csv: Ruta para guardar el CSV con caracteristicas
        bins: Numero de bins para histogramas
        distance: Distancia para GLCM
        
    Returns:
        df: DataFrame con caracteristicas extraidas
    """
    print("=" * 60)
    print("  PROCESAMIENTO DE IMAGENES CAPTURADAS")
    print("=" * 60)
    print(f"\nDirectorio de capturas: {captures_dir}")
    
    # Obtener lista de imagenes
    image_files = [f for f in os.listdir(captures_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("\nNo se encontraron imagenes en el directorio.")
        return None
    
    print(f"Imagenes encontradas: {len(image_files)}")
    
    features_list = []
    filenames = []
    
    for img_name in image_files:
        img_path = os.path.join(captures_dir, img_name)
        print(f"\nProcesando: {img_name}")
        
        try:
            # 1. Cargar imagen
            img = load_image(img_path)
            print(f"  - Imagen cargada: {img.shape}")
            
            # 2. Normalizar pixeles
            img = normalize_image(img)
            print(f"  - Pixeles normalizados a [0, 1]")
            
            # 3. Convertir de vuelta a uint8 para procesamiento
            img = (img * 255).astype(np.uint8)
            
            # 4. Redimensionar al tamaño estandar
            img_resized = resize_image(img, TARGET_SIZE)
            print(f"  - Redimensionada a: {img_resized.shape}")
            
            # 5. Extraer caracteristicas
            features, mask = extract_all_features(img_resized, None, bins, distance)
            print(f"  - Caracteristicas extraidas: {len(features)}")
            
            features_list.append(features)
            filenames.append(img_name)
            
        except Exception as e:
            print(f"  ERROR procesando {img_name}: {str(e)}")
            continue
    
    # Crear DataFrame
    if not features_list:
        print("\nNo se pudieron extraer caracteristicas de ninguna imagen.")
        return None
    
    df = pd.DataFrame(features_list)
    df.insert(0, 'filename', filenames)
    
    # Guardar CSV
    df.to_csv(output_csv, index=False)
    print(f"\n{'-' * 60}")
    print(f"Caracteristicas guardadas en: {output_csv}")
    print(f"Total de imagenes procesadas: {len(df)}")
    print(f"Total de caracteristicas por imagen: {len(df.columns) - 1}")
    print(f"{'-' * 60}")
    
    return df


def predict_with_model(features_df, model_path, mapping_path, predictions_csv):
    """
    Carga el modelo Random Forest y genera predicciones.
    
    Args:
        features_df: DataFrame con caracteristicas
        model_path: Ruta al modelo .pkl
        mapping_path: Ruta al mapeo de etiquetas .pkl
        predictions_csv: Ruta para guardar las predicciones
        
    Returns:
        predictions_df: DataFrame con predicciones
    """
    print("\n" + "=" * 60)
    print("  GENERACION DE PREDICCIONES")
    print("=" * 60)
    
    # Verificar que el modelo existe
    if not os.path.exists(model_path):
        print(f"\nERROR: Modelo no encontrado en {model_path}")
        print("Asegúrate de haber entrenado el modelo primero.")
        return None
    
    # Cargar modelo
    print(f"\nCargando modelo desde: {model_path}")
    model = joblib.load(model_path)
    print("Modelo cargado correctamente")
    
    # Cargar mapeo de etiquetas
    if os.path.exists(mapping_path):
        print(f"Cargando mapeo de etiquetas desde: {mapping_path}")
        label_mapping = joblib.load(mapping_path)
        print(f"Mapeo cargado: {len(label_mapping)} clases")
    else:
        print(f"ADVERTENCIA: No se encontro mapeo de etiquetas en {mapping_path}")
        label_mapping = {}
    
    # Preparar features para prediccion
    filenames = features_df['filename'].values
    X = features_df.drop(columns=['filename'])
    
    print(f"\nGenerando predicciones para {len(X)} imagenes...")
    
    # Predecir
    predictions_encoded = model.predict(X)
    probabilities = model.predict_proba(X)
    confidences = np.max(probabilities, axis=1)
    
    # Decodificar etiquetas
    predictions = []
    for pred in predictions_encoded:
        if label_mapping and pred in label_mapping:
            predictions.append(label_mapping[pred])
        else:
            predictions.append(f"Class_{pred}")
    
    # Crear DataFrame con resultados
    results_df = pd.DataFrame({
        'filename': filenames,
        'predicted_label': predictions,
        'predicted_class': predictions_encoded,
        'confidence': confidences
    })
    
    # Guardar predicciones
    results_df.to_csv(predictions_csv, index=False)
    
    print(f"\n{'-' * 60}")
    print("RESULTADOS:")
    print(f"{'-' * 60}")
    for _, row in results_df.iterrows():
        print(f"{row['filename']:30} -> {row['predicted_label']:20} (confianza: {row['confidence']:.2%})")
    
    print(f"\n{'-' * 60}")
    print(f"Predicciones guardadas en: {predictions_csv}")
    print(f"{'-' * 60}")
    
    return results_df


# ============================================================================
# FUNCION PRINCIPAL
# ============================================================================

def main():
    """
    Funcion principal que ejecuta todo el pipeline.
    """
    print("\n" + "=" * 60)
    print("  PIPELINE DE PREDICCION DE FRUTAS CAPTURADAS")
    print("=" * 60)
    
    # Verificar que existe el directorio de capturas
    if not os.path.exists(CAPTURES_DIR):
        print(f"\nERROR: Directorio de capturas no encontrado: {CAPTURES_DIR}")
        return
    
    # Paso 1: Procesar imagenes y extraer caracteristicas
    features_df = process_captures(
        captures_dir=CAPTURES_DIR,
        output_csv=OUTPUT_CSV,
        bins=8,
        distance=1
    )
    
    if features_df is None:
        print("\nERROR: No se pudieron procesar las imagenes.")
        return
    
    # Paso 2: Generar predicciones con el modelo
    predictions_df = predict_with_model(
        features_df=features_df,
        model_path=MODEL_PATH,
        mapping_path=MAPPING_PATH,
        predictions_csv=PREDICTIONS_CSV
    )
    
    if predictions_df is None:
        print("\nERROR: No se pudieron generar predicciones.")
        return
    
    print("\n" + "=" * 60)
    print("  PROCESO COMPLETADO EXITOSAMENTE")
    print("=" * 60)
    print(f"\nArchivos generados:")
    print(f"  - Caracteristicas: {OUTPUT_CSV}")
    print(f"  - Predicciones:    {PREDICTIONS_CSV}")
    print("")


if __name__ == "__main__":
    main()
