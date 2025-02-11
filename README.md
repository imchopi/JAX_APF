# ADRIÁN PEROGIL FERNÁNDEZ

![image](https://github.com/user-attachments/assets/c2d6b13b-eb20-4f1d-a9fd-5d6e44615fba)
![image](https://github.com/user-attachments/assets/a8f7318d-2ba4-43e1-b529-b93dc4efaec1)

## Qué es JAX

JAX es una librería de Python desarrollada por Google, diseñada para acelerar la investigación en machine learning y computación científica. Combina la facilidad de uso de NumPy con la capacidad de ejecución en hardware acelerado (GPUs y TPUs). JAX es especialmente conocido por su enfoque en la diferenciación automática, la compilación Just-In-Time (JIT) y la vectorización automática, lo que lo convierte en una herramienta poderosa para tareas de alto rendimiento.

### Características principales de JAX
**Diferenciación automática:**

JAX permite calcular gradientes de funciones de manera automática y eficiente, lo que es esencial para entrenar modelos de machine learning.

**Compilación Just-In-Time (JIT):**

Con jit, JAX compila funciones Python para ejecutarlas de manera más rápida en hardware acelerado.
Esto es útil para optimizar el rendimiento en tareas intensivas.

**Vectorización automática:**

vmap permite vectorizar funciones, lo que facilita el procesamiento de lotes de datos sin necesidad de bucles explícitos.

**Compatibilidad con NumPy:**

JAX ofrece una API casi idéntica a NumPy, lo que facilita su adopción para usuarios familiarizados con esta librería.

**Aceleración en hardware:**

JAX puede ejecutar operaciones en GPUs y TPUs sin necesidad de cambios en el código, lo que lo hace ideal para aplicaciones de alto rendimiento.
---

## Comparación con TensorFlow y PyTorch

| Característica       | JAX                          | TensorFlow                   | PyTorch                     |
|----------------------|------------------------------|------------------------------|-----------------------------|
| Facilidad de uso     | Alto (similar a NumPy)       | Medio                        | Alto                        |
| Hardware acelerado   | Excelente (GPUs/TPUs)        | Bueno                        | Bueno                       |
| Madurez              | En crecimiento               | Muy maduro                  | Maduro                      |
| Ecosistema           | Pequeño pero en expansión    | Muy amplio                  | Amplio                      |
| Investigación        | Ideal                        | Bueno                        | Excelente                   |

### Ventajas de JAX frente a TensorFlow y PyTorch
**Flexibilidad:** JAX es más ligero y permite un control más fino sobre las operaciones, lo que lo hace ideal para investigación.

**Rendimiento:** Gracias a jit y vmap, JAX puede ser más rápido en tareas específicas, especialmente en hardware acelerado.

**Simplicidad:** Su API similar a NumPy lo hace más accesible para usuarios que ya están familiarizados con NumPy.

### Desventajas de JAX frente a TensorFlow y PyTorch
**Ecosistema:** JAX tiene un ecosistema más pequeño en comparación con TensorFlow y PyTorch, aunque está creciendo rápidamente.

**Madurez:** TensorFlow y PyTorch tienen una comunidad más grande y más herramientas listas para producción.

---

## Ecosistema de JAX

### Librerías basadas en JAX
**Flax:**

Un framework de alto nivel para construir y entrenar modelos de machine learning.

Ideal para redes neuronales y tareas de deep learning.

**Haiku:**

Desarrollado por DeepMind, es una librería para construir redes neuronales de manera modular.

**Optax:**

Proporciona optimizadores y herramientas para entrenamiento de modelos.

Incluye algoritmos como SGD, Adam, y más.

**RLax:**

Enfocado en aprendizaje por refuerzo, proporciona componentes para construir algoritmos como Q-learning o policy gradients.

**Chex:**

Herramientas para testing y debugging de código en JAX.

### Herramientas que integran con JAX
**TensorFlow Datasets:** Para cargar y preprocesar conjuntos de datos.

**JAXline:** Para entrenamiento distribuido en JAX.

**Equinox:** Combina JAX con PyTorch-like APIs para mayor flexibilidad.

---

## Ejemplo Práctico: Detección de Objetos con YOLO y JAX

En este proyecto, combinamos **YOLOv8** (un modelo de detección de objetos) con **JAX** para procesar una imagen y resaltar los objetos detectados. El código está en el archivo `app.py`.

---

### Descripción del Proyecto

#### 1. **Detección de objetos**:
   - Usamos YOLOv8 para detectar objetos en una imagen.
   - YOLO devuelve las coordenadas de las cajas (bounding boxes), las clases de los objetos y las puntuaciones de confianza.

#### 2. **Procesamiento con JAX**:
   - Convertimos la imagen a un array de JAX para realizar operaciones eficientes.
   - Usamos JAX para resaltar los objetos detectados (dibujar marcos y etiquetas).

#### 3. **Visualización**:
   - Mostramos la imagen original y la imagen con los objetos detectados.

---

## Cómo Ejecutar el Código

1. Clona el repositorio:
   ```bash
   git clone https://github.com/tu_usuario/jax-actividad.git
   cd jax-actividad
   ```

2. Instala las dependencias:
   ```bash
    pip install -r requirements.txt
   ```

3. Ejecutar el script:
   ```bash
    python app.py
   ```

## ¿Qué hice con JAX?

1. Conversión de la imagen a un array de JAX:
Convertí la imagen (que originalmente es un array de NumPy) a un array de JAX usando:
    ```bash
    image_jax = jnp.array(image)
    ```

2. Uso de JAX para operaciones numéricas:
Aunque en este ejemplo no se realizan operaciones numéricas complejas (como transformaciones o diferenciación automática), la conversión a JAX permite que, en futuras extensiones del código, se puedan aplicar operaciones eficientes en hardware acelerado.

3. Conversión de vuelta a NumPy:
Después de realizar las operaciones de dibujo con OpenCV, convertí la imagen de vuelta a un array de JAX:
    ```bash
    return jnp.array(image_np)
    ```

## Bibliografía

A continuación, se listan los recursos y referencias utilizados para este proyecto:

1. **Documentación oficial de JAX**:
   - Enlace: [https://docs.jax.dev/en/latest/](https://docs.jax.dev/en/latest/)
   - Descripción: La documentación oficial de JAX proporciona una guía completa sobre cómo utilizar la librería, incluyendo ejemplos, tutoriales y referencias de la API.

2. **DeepSeek**:
   - Descripción: DeepSeek es una herramienta de búsqueda y documentación especializada en inteligencia artificial y machine learning. Proporciona recursos adicionales y ejemplos prácticos para trabajar con JAX y otras tecnologías relacionadas.

3. **YOLOv8 Documentation**:
   - Enlace: [https://docs.ultralytics.com](https://docs.ultralytics.com)
   - Descripción: Documentación oficial de YOLOv8, que incluye guías de instalación, uso y ejemplos de detección de objetos.

4. **Flax Documentation**:
   - Enlace: [https://github.com/google/flax](https://github.com/google/flax)
   - Descripción: Repositorio oficial de Flax, un framework de alto nivel para construir modelos de machine learning con JAX.

5. **Optax Documentation**:
   - Enlace: [https://github.com/deepmind/optax](https://github.com/deepmind/optax)
   - Descripción: Repositorio oficial de Optax, una librería para optimización en JAX.

6. **OpenCV Documentation**:
   - Enlace: [https://docs.opencv.org](https://docs.opencv.org)
   - Descripción: Documentación oficial de OpenCV, utilizada para el procesamiento de imágenes y dibujo de marcos y etiquetas.

7. **Matplotlib Documentation**:
   - Enlace: [https://matplotlib.org/stable/contents.html](https://matplotlib.org/stable/contents.html)
   - Descripción: Documentación oficial de Matplotlib, utilizada para la visualización de imágenes.

---
