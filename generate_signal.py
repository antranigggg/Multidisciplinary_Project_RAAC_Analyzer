
import numpy as np

def generate_signal(label):
    if label == 1:
        return np.random.normal(2000, 600, 100)
    else:
        return np.random.normal(3500, 150, 100)
