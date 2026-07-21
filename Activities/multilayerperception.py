import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Input data (AND gate)
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

# Output data
y = np.array([0,0,0,1])

# Multi-Layer Perceptron Model
model = Sequential([
    Dense(4, activation='relu', input_shape=(2,)),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=100, verbose=0)

# Predict the output
print("Predictions:")
print(model.predict(X))