from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load your model
model = joblib.load("crop_model99.pkl")

@app.route('/')
def home():
    return "Crop Recommendation API Running 🌱"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Expected input order
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
            "recommended_crop": prediction[0]
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })

if __name__ == "__main__":
    app.run()