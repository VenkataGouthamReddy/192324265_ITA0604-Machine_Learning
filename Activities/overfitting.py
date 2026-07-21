from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

print("Problem: Large model may cause overfitting.")
print("Solution: Add Dropout to reduce overfitting.\n")

model = Sequential([
    Dense(128, activation="relu", input_shape=(10,)),
    Dropout(0.5),            # Turns off 50% of neurons during training

    Dense(64, activation="relu"),
    Dropout(0.3),            # Turns off 30% of neurons during training

    Dense(1, activation="sigmoid")
])

print("Model created successfully.")
print("Dropout reduces overfitting by preventing the model from memorizing the training data.")