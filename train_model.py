import sys
import os
# Adds the root directory (MDP) to the system path so it can find 'simulation'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import tensorflow as tf
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib

# Import local utilities from the simulation directory
# Ensure your python path includes the root directory
from simulation.generate_signal import generate_signal
from simulation.feature_extraction import extract_features

def train_and_save_model():
    # 1. Generate Synthetic Training Data
    X = []
    y = []
    
    print("Generating training data...")
    for _ in range(1000):
        label = np.random.choice([0, 1])
        signal = generate_signal(label)
        features = extract_features(signal)
        X.append(features)
        y.append(label)
    
    X = np.array(X)
    y = np.array(y)

    # 2. Preprocess and Scale Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler for use in the simulation
    joblib.dump(scaler, 'scaler.pkl')
    print("Scaler saved as scaler.pkl")

    # 3. Build the Neural Network
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 4. Train the Model
    print("Training model...")
    model.fit(X_scaled, y, epochs=20, batch_size=32, verbose=0)

    # 5. Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # 6. Save the TFLite model
    with open('raac_model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print("Model saved as raac_model.tflite")

if __name__ == "__main__":
    # Ensure the script runs relative to the model directory
    train_and_save_model()