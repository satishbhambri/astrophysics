import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Synthetic solar activity data: 1000 sequences, 30 time steps each
num_samples = 1000
timesteps = 30


X = np.zeros((num_samples, timesteps, 1))
y = np.zeros(num_samples)
for i in range(num_samples):
    base = np.sin(np.linspace(0, 3 * np.pi, timesteps + 1)) + np.random.normal(0, 0.1, timesteps + 1)
    X[i, :, 0] = base[:-1]
    y[i] = base[-1] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = models.Sequential([
    layers.LSTM(32, input_shape=(timesteps, 1)),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
mse = model.evaluate(X_test, y_test)
print(f"Test MSE: {mse:.4f}")
