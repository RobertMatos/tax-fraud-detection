from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

base_path = os.path.dirname(__file__)
models_path = os.path.join(base_path, "..", "models")

model = joblib.load(os.path.join(models_path, "random_forest_model.pkl"))
scaler = joblib.load(os.path.join(models_path, "scaler.pkl"))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        
        if 'features' not in data or len(data['features']) != 15:
            return jsonify({'error': 'O JSON deve conter exatamente 15 features'}), 400
        
        features = np.array(data['features']).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        
        status = "valid" if prediction[0] == 1 else "not valid"
        return jsonify({'prediction': int(prediction[0]), 'status': status})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
