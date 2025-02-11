# ADRIÁN PEROGIL FERNÁNDEZ

## Qué es JAX

**JAX** es una librería de Python diseñada para acelerar la investigación en machine learning y computación científica. Combina la facilidad de uso de NumPy con la capacidad de ejecución en hardware acelerado (GPUs y TPUs). Sus principales características incluyen:

- **Transformaciones automáticas**: `grad`, `jit`, `vmap`.
- **Compatibilidad con NumPy**: API familiar para usuarios de NumPy.
- **Aceleración en hardware**: Soporte nativo para GPUs y TPUs.
- **Diferenciación automática**: Ideal para optimización y entrenamiento de modelos.

---

## Comparación con TensorFlow y PyTorch

| Característica       | JAX                          | TensorFlow                   | PyTorch                     |
|----------------------|------------------------------|------------------------------|-----------------------------|
| Facilidad de uso     | Alto (similar a NumPy)       | Medio                        | Alto                        |
| Hardware acelerado   | Excelente (GPUs/TPUs)        | Bueno                        | Bueno                       |
| Madurez              | En crecimiento               | Muy maduro                  | Maduro                      |
| Ecosistema           | Pequeño pero en expansión    | Muy amplio                  | Amplio                      |
| Investigación        | Ideal                        | Bueno                        | Excelente                   |

---

## Ecosistema de JAX

JAX cuenta con un ecosistema en crecimiento, que incluye:

- **Librerías basadas en JAX**:
  - **Haiku**: Para construir redes neuronales.
  - **Flax**: Framework de alto nivel para ML.
  - **Optax**: Para optimización.
  - **RLax**: Para aprendizaje por refuerzo.
- **Herramientas integradas**:
  - TensorFlow Datasets.
  - JAXline (entrenamiento distribuido).

---

## Ejemplo Práctico

En el cuaderno Jupyter [`ejemplo_jax.ipynb`](ejemplo_jax.ipynb) se implementa un ejemplo sencillo de **regresión lineal** utilizando JAX. El código incluye:

1. Definición del modelo.
2. Cálculo de gradientes con `grad`.
3. Optimización con `optax`.
4. Uso de `jit` para acelerar el entrenamiento.

---

## Cómo Ejecutar el Código

1. Clona el repositorio:
   ```bash
   git clone https://github.com/tu_usuario/jax-actividad.git
   cd jax-actividad
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