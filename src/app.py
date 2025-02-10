from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

base_path = os.path.dirname(__file__)
models_path = os.path.join(base_path, "..", "models")

model_path = os.path.join(models_path, "random_forest_model.pkl")
scaler_path = os.path.join(models_path, "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
