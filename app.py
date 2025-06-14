from flask import Flask, request, jsonify
import numpy as np
import pickle
import os

app = Flask(__name__)

# โหลดโมเดลที่ train ไว้
model_path = os.path.join(os.path.dirname(__file__), "trained_model.sav")
with open(model_path, "rb") as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return "Diabetes Prediction API พร้อมใช้งาน!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        # ดึงค่า features
        input_features = np.array([list(data.values())])
        prediction = model.predict(input_features)[0]
        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
