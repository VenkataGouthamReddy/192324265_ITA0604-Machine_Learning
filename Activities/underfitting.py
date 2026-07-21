from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

print("Problem: Model is too small and may underfit.")
print("Solution: Use more neurons, hidden layers, and train for more epochs.\n")

# Improved model to reduce underfitting
model = Sequential([
    Dense(64, activation="relu", input_shape=(10,)),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("Model created successfully.")
print("Train the model with more epochs.")
# Example:
# model.fit(X_train, y_train, epochs=50)