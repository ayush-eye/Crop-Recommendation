from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("crop_model99.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    features = [
        data['N'],
        data['P'],
        data['K'],
        data['temperature'],
        data['humidity'],
        data['ph'],
        data['rainfall']
    ]

    final_input = np.array([features])

    prediction = model.predict(final_input)

    return jsonify({
        "recommended_crop": str(prediction[0])
    })