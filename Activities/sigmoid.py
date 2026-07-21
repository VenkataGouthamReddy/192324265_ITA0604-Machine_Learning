import numpy as np
# Input values
x = np.array([-5, -2, 0, 2, 5])
# Apply Sigmoid activation function
sigmoid_output = 1 / (1 + np.exp(-x))
print("Input Values:")
print(x)
print("\nSigmoid Output:")
print(sigmoid_output)
