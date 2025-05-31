# gravitational_wave_classification.py

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split


num_samples = 1000
timesteps = 1024
num_classes = 2  


X = np.random.normal(0, 1, (num_samples, timesteps, 1))
y = np.zeros(num_samples, dtype=int)
for i in range(num_samples // 2):
    # Inject a "chirp" signal for positive class
    t = np.linspace(0, 1, timesteps)
    X[i, :, 0] += 0.5 * np.sin(50 * t**2 * np.pi)
    y[i] = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = models.Sequential([
    layers.Conv1D(16, 5, activation='relu', input_shape=(timesteps, 1)),
    layers.MaxPooling1D(2),
    layers.Conv1D(32, 5, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
loss, acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {acc:.2f}")
