import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore

# Cargar el archivo de Excel
File = 'well-8.xlsx'  # Reemplaza con la ruta de tu archivo
data = pd.read_excel(File)

# Escalar la columna 'Qg'
scaler = MinMaxScaler(feature_range=(0, 1))
data['Qg_scaled'] = scaler.fit_transform(data[['Qg']])

# Definir el porcentaje para la división de datos
train_size = int(len(data) * 0.4)  # 40% para entrenamiento, 60% para prueba

# Dividir los datos en conjuntos de entrenamiento y prueba
train_data = data['Qg_scaled'].values[:train_size]
test_data = data['Qg_scaled'].values[train_size:]

# Definir la longitud de la secuencia (ventana de tiempo)
seq_length = 10

# Función para crear secuencias de datos
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Crear secuencias de entrenamiento y prueba
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Dar forma a X_train y X_test para que sean compatibles con el modelo LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print(f"Forma de X_train: {X_train.shape}")
print(f"Forma de y_train: {y_train.shape}")
print(f"Forma de X_test: {X_test.shape}")
print(f"Forma de y_test: {y_test.shape}")

# Definir el modelo LSTM con función de activación lineal en la salida
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))  # Capa LSTM con ReLU
model.add(Dense(1, activation='linear'))  # Capa de salida con activación lineal

# Compilar el modelo con el optimizador Adam y la función de pérdida MSE
model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# Graficar la pérdida de entrenamiento y validación
plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.title('Pérdida de entrenamiento y validación a lo largo de las épocas')
plt.legend()
plt.show()

# Evaluar el modelo en el conjunto de prueba
test_loss = model.evaluate(X_test, y_test, verbose=1)
print(f'Pérdida en el conjunto de prueba: {test_loss}')

# Predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Desescalar los datos
train_descaled = scaler.inverse_transform(train_data.reshape(-1, 1))
test_descaled = scaler.inverse_transform(test_data.reshape(-1, 1))
y_pred_descaled = scaler.inverse_transform(y_pred)

# Crear una gráfica con datos de entrenamiento, prueba y predicciones
plt.figure(figsize=(14, 6))

# Graficar los datos de entrenamiento
plt.plot(range(len(train_descaled)), train_descaled, label='Datos de Entrenamiento', color='blue')

# Graficar los datos de prueba
plt.plot(range(len(train_descaled), len(train_descaled) + len(test_descaled)), test_descaled, label='Datos de Prueba', color='green')

# Graficar las predicciones en el conjunto de prueba
plt.plot(range(len(train_descaled), len(train_descaled) + len(y_pred_descaled)), y_pred_descaled, label='Predicciones', color='red')

# Configuración de la gráfica
plt.xlabel('Índice de Muestra')
plt.ylabel('Producción de Gas (Qg)')
plt.title('Datos de Entrenamiento, Prueba y Predicciones')
plt.legend()
plt.show()


