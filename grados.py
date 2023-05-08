import numpy as np
import tensorflow as tf

def calcular(grados_celsius):
    # Datos de entrada (grados Celsius) y salida esperada (grados Fahrenheit)
    celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
    fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

    # Definición de las capas de la red neuronal
    neurona_1 = tf.keras.layers.Dense(units=3, input_shape=[1])
    neurona_2 = tf.keras.layers.Dense(units=3)
    salida = tf.keras.layers.Dense(units=1)
    modelo = tf.keras.Sequential([neurona_1, neurona_2, salida])
    # Configuración del modelo
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(0.01),
        loss='mean_squared_error'
    )
    print("Comenzando entrenamiento...")
    # Entrenamiento del modelo
    historial = modelo.fit(celsius, fahrenheit, epochs=300, verbose=False)
    print("Modelo entrenado!")
    # Realización de la predicción con el valor de grados Celsius proporcionado
    resultado = modelo.predict([grados_celsius])
    print(f"Predicción: {grados_celsius} grados Celsius son {resultado} grados Fahrenheit!")

if __name__ == "__main__":
    grados_celsius = 100.0  # Valor de grados Celsius a pasar como argumento
    calcular(grados_celsius)

