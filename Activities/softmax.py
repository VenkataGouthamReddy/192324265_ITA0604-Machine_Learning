import numpy as np
marks = np.array([85, 70, 60])
softmax = np.exp(marks) / np.sum(np.exp(marks))
print("Scores :", marks)
print("Softmax:", softmax)
