import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Dataset
# Features: [Study Hours, Attendance]
X = np.array([
    [2,60],
    [3,65],
    [4,70],
    [5,75],
    [6,80],
    [7,85],
    [8,90],
    [5,68],
    [6,72],
    [3,78]
], dtype=float)

# Target: Pass(1) / Fail(0)
y = np.array([0,0,0,1,1,1,1,0,1,0])

# Build Deep Neural Network
model = Sequential([
    Dense(6, activation='relu', input_shape=(2,)),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(
    X,
    y,
    epochs=10,
    batch_size=2,
    verbose=1
)