
import numpy as np
import joblib
import tensorflow as tf
from generate_signal import generate_signal
from feature_extraction import extract_features

interpreter = tf.lite.Interpreter(model_path="../model/raac_model.tflite")
interpreter.allocate_tensors()

scaler = joblib.load("../model/scaler.pkl")

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

for i in range(5):
    label = np.random.choice([0,1])
    signal = generate_signal(label)
    features = extract_features(signal)

    X = scaler.transform([features])

    interpreter.set_tensor(input_details[0]['index'], X.astype('float32'))
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])[0][0]

    print(f"Prediction: {'Abnormal' if output > 0.5 else 'Normal'} ({output:.2f})")
