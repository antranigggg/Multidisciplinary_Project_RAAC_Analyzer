
import numpy as np

def extract_features(signal):
    return [
        np.mean(signal),
        np.std(signal),
        np.min(signal),
        np.max(signal),
        np.std(signal)/np.mean(signal)
    ]
