# Optimización de Funciones con Streamlit y Jupyter

Este repositorio incluye dos enfoques para optimización de funciones: una aplicación interactiva con **Streamlit** y un estudio más profundo utilizando **Jupyter Notebook**. Ambas implementaciones prueban métodos de optimización como **Gradient Descent**, **Stochastic Gradient Descent**, **RMSPROP** y **Adam** para estimar el punto mínimo de una función multivariable.

## Streamlit Application

### Descripción

La aplicación **Streamlit** permite al usuario interactuar con la visualización de una función de dos variables, modificar los límites de `x` y `y` para observar el gráfico, y luego seleccionar el método de optimización. El objetivo es estimar el punto mínimo utilizando uno de los siguientes métodos:

- **Gradient Descent**
- **Stochastic Gradient Descent**
- **RMSPROP**
- **Adam**

### Características

- **Configuración**: Ajusta los límites de `x` y `y` para graficar la función `f(x, y) = -sin(sqrt(x^2 + y^2))`.
- **Optimización**: Selecciona un método de optimización y ajusta los parámetros correspondientes (tasa de aprendizaje, número de épocas, etc.).
- **Resultado**: Calcula el punto mínimo estimado y la distancia al origen `(0,0)`.

## Jupyter Notebook

### Descripción

El **Jupyter Notebook** incluido realiza un estudio de los métodos de optimización mencionados anteriormente. Cada método se ejecuta con 10,000 iteraciones para comparar la efectividad en estimar el punto mínimo. El notebook también calcula la distancia promedio al origen `(0,0)` para cada método y muestra una tabla de frecuencias que indica cuál método fue el más efectivo.
