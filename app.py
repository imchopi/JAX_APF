# app.py
# Importar las librerías necesarias
import cv2  # OpenCV para manipulación de imágenes y dibujo
import jax.numpy as jnp  # JAX para operaciones numéricas eficientes
import numpy as np  # NumPy para operaciones adicionales
from ultralytics import YOLO  # YOLOv8 para detección de objetos
import matplotlib.pyplot as plt  # Matplotlib para visualización

# 1. Cargar el modelo YOLOv8 preentrenado
# Usamos YOLOv8 nano (el más ligero y rápido)
model = YOLO("yolov8n.pt")

# 2. Cargar una imagen de ejemplo
# Cambia "ejemplo.jpg" por la ruta de tu imagen
image_path = "ejemplo.jpg"
image = cv2.imread(image_path)  # Leer la imagen con OpenCV
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB (para visualización correcta)

# 3. Realizar la detección de objetos con YOLO
# El modelo procesa la imagen y devuelve los resultados
results = model(image)

# 4. Obtener las cajas (bounding boxes), clases y puntuaciones
# Las cajas son las coordenadas de los rectángulos que enmarcan los objetos detectados
boxes = results[0].boxes.xyxy.cpu().numpy()  # Coordenadas de las cajas (x1, y1, x2, y2)
classes = results[0].boxes.cls.cpu().numpy()  # Clases de los objetos (índices)
scores = results[0].boxes.conf.cpu().numpy()  # Puntuaciones de confianza (0 a 1)

# 5. Convertir la imagen a un array de JAX
# JAX permite operaciones numéricas eficientes en hardware acelerado (GPU/TPU)
image_jax = jnp.array(image)  # Convertir la imagen de NumPy a JAX

# 6. Función para dibujar marcos y etiquetas usando JAX y OpenCV
def dibujar_marcos_y_etiquetas(image_jax, boxes, classes, scores):
    # Convertir la imagen de JAX a NumPy para usar OpenCV (JAX no tiene funciones de dibujo)
    image_np = np.array(image_jax)
    
    # Iterar sobre cada objeto detectado
    for box, cls, score in zip(boxes, classes, scores):
        x1, y1, x2, y2 = box.astype(int)  # Coordenadas del rectángulo
        label = f"{model.names[int(cls)]} {score:.2f}"  # Etiqueta: nombre del objeto + puntuación
        
        # 7. Dibujar el marco (borde) del rectángulo
        # Usamos OpenCV para dibujar un rectángulo rojo alrededor del objeto
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (255, 0, 0), 2)  # (255, 0, 0) = rojo, 2 = grosor
        
        # 8. Dibujar la etiqueta encima del rectángulo
        font = cv2.FONT_HERSHEY_SIMPLEX  # Fuente del texto
        font_scale = 0.8  # Tamaño de la fuente
        thickness = 2  # Grosor de la fuente (negrita)
        
        # Calcular el tamaño del texto para ajustar su posición
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        
        # Posición del texto: encima del rectángulo
        text_x = x1  # Alineado a la izquierda del rectángulo
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20  # Ajustar si está cerca del borde superior
        
        # Dibujar el texto en blanco y en negrita
        cv2.putText(
            image_np,  # Imagen donde se dibuja
            label,  # Texto a dibujar
            (text_x, text_y),  # Posición del texto
            font,  # Fuente
            font_scale,  # Tamaño de la fuente
            (255, 255, 255),  # Color del texto (blanco)
            thickness,  # Grosor de la fuente
            cv2.LINE_AA  # Tipo de línea (suavizada)
        )
    
    # Convertir la imagen de nuevo a JAX para mantener la coherencia
    return jnp.array(image_np)

# 9. Aplicar la función para dibujar marcos y etiquetas
image_con_marcos = dibujar_marcos_y_etiquetas(image_jax, boxes, classes, scores)

# 10. Convertir la imagen de JAX a NumPy para visualización
image_con_marcos_np = np.array(image_con_marcos)

# 11. Mostrar la imagen original y la imagen con marcos y etiquetas
plt.figure(figsize=(12, 6))  # Crear una figura de 12x6 pulgadas

# Imagen original
plt.subplot(1, 2, 1)  # Subplot 1 de 2
plt.title("Imagen Original")
plt.imshow(image)
plt.axis("off")  # Ocultar ejes

# Imagen con objetos detectados
plt.subplot(1, 2, 2)  # Subplot 2 de 2
plt.title("Objetos Detectados")
plt.imshow(image_con_marcos_np)
plt.axis("off")  # Ocultar ejes

plt.show()  # Mostrar la figura