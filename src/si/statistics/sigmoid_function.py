import numpy as np

def sigmoid_function(X):
    
    return 1 / (1 + np.exp(-X))