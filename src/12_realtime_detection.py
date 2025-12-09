import cv2
import numpy as np

def segment_fruit_hsv(img, hsv_ranges=None):
    """
    Segmenta frutas usando rangos HSV adaptativos.
    
    Args:
        img: numpy array - Imagen BGR de entrada
        hsv_ranges: list - Lista de tuplas (lower, upper) con rangos HSV. 
                          Si None, usa rangos por defecto para frutas comunes.
    
    Returns:
        mask: numpy array - Máscara binaria de la fruta detectada
    """
    # Convertir a HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Rangos HSV por defecto para diferentes tipos de frutas
    if hsv_ranges is None:
        # Rangos amplios para capturar diferentes frutas
        # Naranjas, manzanas rojas, plátanos amarillos, etc.
        ranges = [
            # Rango para frutas rojas/naranjas (manzanas rojas, naranjas)
            (np.array([0, 50, 50]), np.array([25, 255, 255])),
            # Rango para frutas amarillas (plátanos, limones)
            (np.array([20, 50, 50]), np.array([35, 255, 255])),
            # Rango para frutas verdes (manzanas verdes)
            (np.array([35, 50, 50]), np.array([85, 255, 255])),
        ]
    else:
        ranges = hsv_ranges
    
    # Combinar todas las máscaras
    combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    
    for lower, upper in ranges:
        mask = cv2.inRange(hsv, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    return combined_mask


def segment_fruit_advanced(img):
    """
    Segmenta frutas usando una combinación de técnicas:
    1. Eliminación de fondo (detección de piel)
    2. Segmentación por color HSV
    3. Filtrado por tamaño y forma
    
    Args:
        img: numpy array - Imagen BGR de entrada
    
    Returns:
        mask: numpy array - Máscara binaria de la fruta detectada
        contour: numpy array - Contorno de la fruta más grande
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Estrategia 1: Eliminar fondos claros (paredes, manos, etc.)
    # Detectar píxeles muy claros o muy oscuros (probablemente no fruta)
    light_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 55, 255]))
    dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 50]))
    background_mask = cv2.bitwise_or(light_mask, dark_mask)
    
    # Estrategia 2: Detectar colores de frutas
    fruit_color_mask = segment_fruit_hsv(img)
    
    # Combinar: fruta debe tener color de fruta Y no ser fondo claro/oscuro
    combined_mask = cv2.bitwise_and(fruit_color_mask, cv2.bitwise_not(background_mask))
    
    # Operaciones morfológicas para limpiar la máscara
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return combined_mask, None
    
    # Filtrar por área mínima (eliminar ruido)
    min_area = img.shape[0] * img.shape[1] * 0.01  # Al menos 1% del área de la imagen
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    if not valid_contours:
        return combined_mask, None
    
    # Obtener el contorno más grande (probablemente la fruta)
    largest_contour = max(valid_contours, key=cv2.contourArea)
    
    # Crear máscara solo con el contorno más grande
    final_mask = np.zeros(combined_mask.shape, dtype=np.uint8)
    cv2.drawContours(final_mask, [largest_contour], -1, 255, -1)
    
    return final_mask, largest_contour


def apply_mask_to_image(img, mask, alpha=0.6):
    """
    Aplica la máscara a la imagen con un efecto de overlay.
    
    Args:
        img: numpy array - Imagen original BGR
        mask: numpy array - Máscara binaria
        alpha: float - Transparencia del overlay (0-1)
    
    Returns:
        result: numpy array - Imagen con overlay de la máscara
    """
    # Crear imagen con coloreado de la máscara
    colored_mask = img.copy()
    colored_mask[mask > 0] = [0, 255, 0]  # Verde para la fruta detectada
    
    # Combinar imagen original con máscara coloreada
    result = cv2.addWeighted(img, 1-alpha, colored_mask, alpha, 0)
    
    return result


def draw_contour_info(img, contour, mask):
    """
    Dibuja información del contorno en la imagen.
    
    Args:
        img: numpy array - Imagen donde dibujar
        contour: numpy array - Contorno de la fruta
        mask: numpy array - Máscara binaria
    
    Returns:
        img: numpy array - Imagen con información dibujada
    """
    if contour is None:
        return img
    
    # Dibujar contorno
    cv2.drawContours(img, [contour], -1, (0, 255, 255), 2)
    
    # Calcular área y circularidad
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    
    # Obtener bounding box
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Dibujar información de texto
    info_text = [
        f"Area: {int(area)} px",
        f"Circularity: {circularity:.2f}",
        f"Size: {w}x{h}"
    ]
    
    y_offset = 30
    for i, text in enumerate(info_text):
        cv2.putText(img, text, (x, y - y_offset + i * 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, text, (x, y - y_offset + i * 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    return img


def main():
    """
    Función principal para detección de frutas en tiempo real.
    """
    # Inicializar captura de video
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara")
        return
    
    print("Detección de frutas en tiempo real")
    print("=" * 40)
    print("Instrucciones:")
    print("- Presiona 'q' para salir")
    print("- Presiona 'm' para alternar visualización de máscara")
    print("- Presiona 'i' para alternar visualización de información")
    print("=" * 40)
    
    show_mask = False
    show_info = True
    
    while True:
        # Capturar frame
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame")
            break
        
        # Voltear horizontalmente para efecto espejo (más natural)
        frame = cv2.flip(frame, 1)
        
        # Segmentar fruta
        mask, contour = segment_fruit_advanced(frame)
        
        # Crear visualización
        if show_mask:
            # Mostrar máscara en escala de grises
            display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            # Mostrar imagen con overlay
            display = apply_mask_to_image(frame, mask)
        
        # Agregar información del contorno
        if show_info and contour is not None:
            display = draw_contour_info(display, contour, mask)
        
        # Agregar texto de instrucciones
        cv2.putText(display, "Presiona 'q' para salir", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, "Presiona 'm' para máscara, 'i' para info", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Mostrar frame
        cv2.imshow('Deteccion de Frutas en Tiempo Real', display)
        
        # Manejar teclas
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            show_mask = not show_mask
        elif key == ord('i'):
            show_info = not show_info
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    print("\nDetección finalizada")


if __name__ == "__main__":
    main()

