#!/usr/bin/env python3
"""
Sistema simplificado de aislamiento de frutas en tiempo real.
Detecta y aisla frutas que sostienes con las manos usando la webcam.
"""

import cv2
import numpy as np


# ============================================================================
# DETECCION DE PIEL (MANOS/CARA)
# ============================================================================

def detect_skin(frame):
    """
    Detecta areas de color piel (manos, brazos, cara).
    
    Args:
        frame: Imagen BGR de entrada
        
    Returns:
        Mascara binaria donde las areas de piel son blancas (255)
    """
    # Convertir a espacio de color HSV y YCrCb para mejor deteccion
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    
    # Rangos HSV - mas permisivos
    lower_hsv = np.array([0, 15, 0], dtype=np.uint8)
    upper_hsv = np.array([25, 170, 255], dtype=np.uint8)
    skin_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    # Rangos YCrCb - deteccion complementaria
    lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
    skin_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
    
    # Combinar ambas mascaras
    skin_mask = cv2.bitwise_and(skin_hsv, skin_ycrcb)
    
    # Limpiar ruido
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Dilatar para excluir bien las manos
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skin_mask = cv2.dilate(skin_mask, kernel_dilate, iterations=1)
    
    return skin_mask


# ============================================================================
# DETECCION DE FRUTAS POR COLOR
# ============================================================================

def detect_fruit_colors(frame):
    """
    Detecta colores tipicos de frutas en el espacio HSV.
    Rangos mas amplios para mejor deteccion.
    
    Args:
        frame: Imagen BGR de entrada
        
    Returns:
        Mascara binaria donde las frutas detectadas son blancas (255)
    """
    # Convertir a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Crear mascara vacia
    fruit_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    # Rangos mas amplios y permisivos
    color_ranges = {
        'rojo_1': ([0, 50, 50], [10, 255, 255]),      # Rojos parte 1
        'rojo_2': ([170, 50, 50], [180, 255, 255]),   # Rojos parte 2
        'naranja': ([8, 60, 60], [25, 255, 255]),     # Naranjas
        'amarillo': ([20, 50, 50], [40, 255, 255]),   # Amarillos
        'verde': ([35, 40, 40], [90, 255, 255]),      # Verdes
        'morado': ([120, 40, 40], [160, 255, 255]),   # Morados/azules
    }
    
    # Aplicar cada rango de color
    for color_name, (lower, upper) in color_ranges.items():
        lower_np = np.array(lower, dtype=np.uint8)
        upper_np = np.array(upper, dtype=np.uint8)
        
        mask = cv2.inRange(hsv, lower_np, upper_np)
        fruit_mask = cv2.bitwise_or(fruit_mask, mask)
    
    # Limpiar ruido
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fruit_mask = cv2.morphologyEx(fruit_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    fruit_mask = cv2.morphologyEx(fruit_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return fruit_mask


# ============================================================================
# AISLAMIENTO DE FRUTA
# ============================================================================

def isolate_fruit(frame):
    """
    Aisla la fruta del resto de la imagen (excluyendo manos, cara, fondo).
    
    Args:
        frame: Imagen BGR de entrada
        
    Returns:
        fruit_mask: Mascara binaria de la fruta aislada
        largest_contour: Contorno de la fruta mas grande (o None)
    """
    # 1. Detectar colores de fruta primero
    fruit_color_mask = detect_fruit_colors(frame)
    
    # 2. Detectar areas de piel (manos, cara)
    skin_mask = detect_skin(frame)
    
    # 3. Excluir areas muy oscuras (ropa oscura, sombras)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 40]))
    
    # 4. Excluir areas muy claras (fondo blanco)
    light_mask = cv2.inRange(hsv, np.array([0, 0, 240]), np.array([180, 20, 255]))
    
    # 5. Crear mascara de exclusion total
    exclusion_mask = cv2.bitwise_or(skin_mask, dark_mask)
    exclusion_mask = cv2.bitwise_or(exclusion_mask, light_mask)
    
    # 6. Aislar fruta: color de fruta SIN areas excluidas
    fruit_mask = cv2.bitwise_and(fruit_color_mask, cv2.bitwise_not(exclusion_mask))
    
    # 7. Limpiar mascara final
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fruit_mask = cv2.morphologyEx(fruit_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    fruit_mask = cv2.morphologyEx(fruit_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 8. Encontrar contornos
    contours, _ = cv2.findContours(fruit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return fruit_mask, None
    
    # 9. Filtrar contornos por area minima (mas permisivo)
    min_area = frame.shape[0] * frame.shape[1] * 0.005  # 0.5% del area de la imagen
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    if not valid_contours:
        return fruit_mask, None
    
    # 10. Seleccionar el contorno mas grande (probablemente la fruta)
    largest_contour = max(valid_contours, key=cv2.contourArea)
    
    # 11. Crear mascara final solo con la fruta principal
    final_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(final_mask, [largest_contour], -1, 255, -1)
    
    return final_mask, largest_contour


# ============================================================================
# VISUALIZACION
# ============================================================================

def draw_fruit_overlay(frame, mask, contour):
    """
    Dibuja un overlay sobre la fruta detectada y muestra informacion.
    
    Args:
        frame: Imagen original
        mask: Mascara de la fruta
        contour: Contorno de la fruta
        
    Returns:
        Imagen con overlay
    """
    result = frame.copy()
    
    # Crear overlay verde semi-transparente sobre la fruta
    overlay = frame.copy()
    overlay[mask > 0] = [0, 255, 0]  # Verde
    result = cv2.addWeighted(result, 0.6, overlay, 0.4, 0)
    
    # Si hay un contorno valido, dibujar informacion
    if contour is not None:
        # Dibujar contorno amarillo
        cv2.drawContours(result, [contour], -1, (0, 255, 255), 3)
        
        # Obtener bounding box
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 255), 2)
        
        # Calcular informacion
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        
        # Mostrar informacion
        info_texts = [
            f"Area: {int(area)} px",
            f"Tamano: {w}x{h} px",
            f"Circularidad: {circularity:.2f}"
        ]
        
        # Dibujar textos con sombra para mejor legibilidad
        y_offset = max(30, y - 10)
        for i, text in enumerate(info_texts):
            text_y = y_offset - (len(info_texts) - i - 1) * 25
            
            # Sombra negra
            cv2.putText(result, text, (x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            # Texto blanco
            cv2.putText(result, text, (x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return result


def create_debug_view(frame, fruit_mask, fruit_color_mask, skin_mask):
    """
    Crea una vista de debug con multiples paneles.
    
    Args:
        frame: Imagen original
        fruit_mask: Mascara de la fruta final
        fruit_color_mask: Mascara de colores de fruta (sin exclusiones)
        skin_mask: Mascara de piel
        
    Returns:
        Imagen con vista de debug (4 paneles)
    """
    h, w = frame.shape[:2]
    
    # Crear imagen grande para 4 paneles
    debug_view = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
    
    # Panel 1: Original
    debug_view[0:h, 0:w] = frame
    cv2.putText(debug_view, "Original", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Panel 2: Mascara de COLOR de fruta (antes de exclusiones)
    fruit_color = cv2.cvtColor(fruit_color_mask, cv2.COLOR_GRAY2BGR)
    debug_view[0:h, w:w*2] = fruit_color
    cv2.putText(debug_view, "Colores Fruta", (w + 10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Panel 3: Mascara de piel
    skin_colored = cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2BGR)
    debug_view[h:h*2, 0:w] = skin_colored
    cv2.putText(debug_view, "Piel/Manos", (10, h + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    
    # Panel 4: Fruta FINAL aislada
    fruit_final = cv2.cvtColor(fruit_mask, cv2.COLOR_GRAY2BGR)
    debug_view[h:h*2, w:w*2] = fruit_final
    cv2.putText(debug_view, "Fruta Aislada", (w + 10, h + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return debug_view


# ============================================================================
# FUNCION PRINCIPAL
# ============================================================================

def main():
    """
    Funcion principal para aislamiento de frutas en tiempo real.
    """
    print("=" * 60)
    print("    SISTEMA DE AISLAMIENTO DE FRUTAS EN TIEMPO REAL")
    print("=" * 60)
    print("\nIniciando camara...")
    
    # Inicializar camara
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: No se pudo abrir la camara")
        return
    
    # Configurar resolucion
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Camara iniciada correctamente\n")
    print("CONTROLES:")
    print("  'q' o ESC  - Salir")
    print("  'm'        - Alternar vista de mascara")
    print("  'd'        - Alternar vista de debug (4 paneles)")
    print("  'i'        - Alternar informacion de la fruta")
    print("  ESPACIO    - Capturar imagen de la fruta aislada")
    print("=" * 60 + "\n")
    
    # Estados de visualizacion
    show_mask = False
    show_debug = False
    show_info = True
    
    # Contador para capturas
    capture_count = 0
    
    while True:
        # Capturar frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: No se pudo leer el frame")
            break
        
        # Voltear horizontalmente (efecto espejo)
        frame = cv2.flip(frame, 1)
        
        # Detectar piel (para debug)
        skin_mask = detect_skin(frame)
        
        # Detectar colores de fruta (para debug)
        fruit_color_mask = detect_fruit_colors(frame)
        
        # Aislar fruta
        fruit_mask, contour = isolate_fruit(frame)
        
        # Crear visualizacion segun el modo
        if show_debug:
            # Vista de debug con 4 paneles
            display = create_debug_view(frame, fruit_mask, fruit_color_mask, skin_mask)
            display = cv2.resize(display, (frame.shape[1], frame.shape[0]))
            
        elif show_mask:
            # Solo mostrar la mascara de la fruta
            display = cv2.cvtColor(fruit_mask, cv2.COLOR_GRAY2BGR)
            
        else:
            # Vista normal con overlay
            display = draw_fruit_overlay(frame, fruit_mask, contour if show_info else None)
        
        # Anadir instrucciones en pantalla
        instructions = "Q/ESC: Salir  |  M: Mascara  |  D: Debug  |  I: Info  |  ESPACIO: Capturar"
        
        cv2.putText(display, instructions, (10, display.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Indicador de fruta detectada
        if contour is not None:
            status_text = "FRUTA DETECTADA"
            status_color = (0, 255, 0)
        else:
            status_text = "Buscando fruta..."
            status_color = (0, 165, 255)
        
        cv2.putText(display, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        cv2.putText(display, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Mostrar ventana
        cv2.imshow('Aislamiento de Frutas', display)
        
        # Procesar teclas
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' o ESC
            print("\nSaliendo...")
            break
            
        elif key == ord('m'):  # Alternar mascara
            show_mask = not show_mask
            show_debug = False
            mode = "MASCARA" if show_mask else "NORMAL"
            print(f"Modo: {mode}")
            
        elif key == ord('d'):  # Alternar debug
            show_debug = not show_debug
            show_mask = False
            mode = "DEBUG" if show_debug else "NORMAL"
            print(f"Modo: {mode}")
            
        elif key == ord('i'):  # Alternar info
            show_info = not show_info
            mode = "ACTIVADA" if show_info else "DESACTIVADA"
            print(f"Informacion: {mode}")
            
        elif key == ord(' '):  # ESPACIO - Capturar
            if contour is not None:
                # Crear imagen con solo la fruta sobre fondo blanco
                captured = np.ones_like(frame) * 255
                captured[fruit_mask > 0] = frame[fruit_mask > 0]
                
                # Guardar imagen
                filename = f"fruta_aislada_{capture_count:03d}.png"
                cv2.imwrite(filename, captured)
                capture_count += 1
                print(f"Captura guardada: {filename}")
            else:
                print("No hay fruta detectada para capturar")
    
    # Limpiar
    cap.release()
    cv2.destroyAllWindows()
    print("\nPrograma finalizado correctamente")
    print(f"Total de capturas: {capture_count}")


if __name__ == "__main__":
    main()