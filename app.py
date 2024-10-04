import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Definir la función de pérdida
def loss_func(theta):
    x, y = theta
    R = np.sqrt(x**2 + y**2) + 1e-8  # Evitar división por cero
    return -np.sin(R)

# Definir el gradiente de la función de pérdida
def evaluate_gradient(theta):
    x, y = theta
    R = np.sqrt(x**2 + y**2) + 1e-8
    grad_x = -np.cos(R) * (x / R)
    grad_y = -np.cos(R) * (y / R)
    return np.array([grad_x, grad_y])

# Métodos de optimización

# 1. Gradiente Descendente
def gradient_descent(theta_init, evaluate_gradient, eta, epochs):
    theta = theta_init.copy()
    for _ in range(epochs):
        gradient = evaluate_gradient(theta)
        theta -= eta * gradient
    return theta

# 2. Gradiente Descendente Estocástico (SGD)
def stochastic_gradient_descent(theta_init, data_train, evaluate_gradient, eta, epochs):
    theta = theta_init.copy()
    for _ in range(epochs):
        np.random.shuffle(data_train)
        for _ in data_train:
            gradient = evaluate_gradient(theta)
            theta -= eta * gradient
    return theta

# 3. RMSPROP
def rmsprop(theta_init, data_train, evaluate_gradient, eta, epochs, decay=0.9, epsilon=1e-8):
    theta = theta_init.copy()
    Eg2 = np.zeros_like(theta)
    for _ in range(epochs):
        np.random.shuffle(data_train)
        for _ in data_train:
            gradient = evaluate_gradient(theta)
            Eg2 = decay * Eg2 + (1 - decay) * gradient**2
            theta -= (eta / (np.sqrt(Eg2) + epsilon)) * gradient
    return theta

# 4. Adam
def adam(theta_init, data_train, evaluate_gradient, eta, epochs, beta1=0.9, beta2=0.999, epsilon=1e-8):
    theta = theta_init.copy()
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    t = 0
    for _ in range(epochs):
        np.random.shuffle(data_train)
        for _ in data_train:
            t += 1
            gradient = evaluate_gradient(theta)
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * gradient**2
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            theta -= (eta / (np.sqrt(v_hat) + epsilon)) * m_hat
    return theta

# Función para graficar
def plot_function(x_min, x_max, y_min, y_max):
    X = np.linspace(x_min, x_max, 100)
    Y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = -np.sin(R)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Función f(x, y) = -sin(sqrt(x^2 + y^2))')
    return fig

# Aplicación Streamlit
def main():
    st.title("Optimización de Funciones")
    
    # Crear dos columnas
    col1, col2 = st.columns(2)
    
    # Columna Izquierda: Ingreso de límites y graficación
    with col1:
        st.header("Configuración")
        x_min = st.number_input("Límite inferior de x", value=-6.5)
        x_max = st.number_input("Límite superior de x", value=6.5)
        y_min = st.number_input("Límite inferior de y", value=-6.5)
        y_max = st.number_input("Límite superior de y", value=6.5)
        
        # Graficar la función
        fig = plot_function(x_min, x_max, y_min, y_max)
        st.pyplot(fig)
    
    # Columna Derecha: Selección de método y parámetros
    with col2:
        st.header("Optimización")
        method = st.selectbox("Selecciona el método de optimización", 
                              ["Gradient Descent", "Stochastic Gradient Descent", "RMSPROP", "Adam"])
        
        # Parámetros comunes
        eta = st.number_input("Tasa de aprendizaje", value=0.01)
        epochs = st.number_input("Número de épocas", min_value=1, value=100)
        theta_init_x = st.number_input("Valor inicial de x", value=2.0)
        theta_init_y = st.number_input("Valor inicial de y", value=2.0)
        theta_init = np.array([theta_init_x, theta_init_y])
        
        # Parámetros específicos para RMSPROP y Adam
        if method == "RMSPROP":
            decay = st.number_input("Decaimiento (decay)", value=0.9, format="%.2f")
        if method == "Adam":
            beta1 = st.number_input("Beta1", value=0.9, format="%.2f")
            beta2 = st.number_input("Beta2", value=0.999, format="%.3f")
        
        calculate_button = st.button("Calcular")
        
        if calculate_button:
            # Datos de entrenamiento ficticios
            data_train = [np.array([0, 0])]  # No se usan realmente en la función
            
            if method == "Gradient Descent":
                theta_final = gradient_descent(theta_init, evaluate_gradient, eta, int(epochs))
            elif method == "Stochastic Gradient Descent":
                theta_final = stochastic_gradient_descent(theta_init, data_train, evaluate_gradient, eta, int(epochs))
            elif method == "RMSPROP":
                theta_final = rmsprop(theta_init, data_train, evaluate_gradient, eta, int(epochs), decay=decay)
            elif method == "Adam":
                theta_final = adam(theta_init, data_train, evaluate_gradient, eta, int(epochs), beta1=beta1, beta2=beta2)
            
            distance = np.linalg.norm(theta_final - np.array([0, 0]))
            
            st.success(f"El punto mínimo estimado es:  \nx = {theta_final[0]:.5f}, y = {theta_final[1]:.5f}")
            st.success(f"Distancia al punto (0, 0):  \n{distance:.5f}")

if __name__ == "__main__":
    main()
