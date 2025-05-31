import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Synthetic data: 1000 light curves, 200 time steps each
num_samples = 1000
timesteps = 200
num_classes = 2  # transit/no transit

# Generate synthetic light curves (random noise + dip for transits)
X = np.random.normal(1, 0.01, (num_samples, timesteps, 1))
y = np.zeros(num_samples, dtype=int)
for i in range(num_samples // 2):
    dip_start = np.random.randint(50, 150)
    X[i, dip_start:dip_start+10, 0] -= 0.02  
    y[i] = 1 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = models.Sequential([
    layers.LSTM(32, input_shape=(timesteps, 1)),
    layers.Dense(16, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
loss, acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {acc:.2f}")
