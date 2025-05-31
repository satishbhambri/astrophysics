import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


num_samples = 1000
img_height, img_width = 64, 64
num_classes = 3  # e.g., spiral, elliptical, irregular

X = np.random.rand(num_samples, img_height, img_width, 3)
y = np.random.randint(0, num_classes, num_samples)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))


loss, acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {acc:.2f}")
