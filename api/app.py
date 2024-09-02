from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the diabetes model
diabetes_model_path = 'diabetes_model.sav'
diabetes_model = pickle.load(open(diabetes_model_path, 'rb'))

# Diabetes prediction route
@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    try:
        data = request.json
        required_fields = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing fields in request'}), 400

        features = [float(data[field]) for field in required_fields]
        prediction = diabetes_model.predict([features])

        return jsonify(result=int(prediction[0]))

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
