import numpy as np
# Input values
x = np.array([-5, -2, 0, 3, 7])
# Apply ReLU activation function
relu_output = np.maximum(0, x)
print("Input Values:")
print(x)
print("\nReLU Output:")
print(relu_output)
