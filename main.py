from flask import Flask, request, jsonify
from flask_cors import CORS

import numpy as np
import joblib

app = Flask(__name__)
CORS(app)

rf_model = joblib.load('random_florest_model.sav')
lr_model = joblib.load('logistic_regression_model.sav')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        model = rf_model

        if data['Model'] == 'logistic_regression_model':
            model = lr_model

        features = np.array([
            data['Age'],
            data['BMI'],
            data['HighBP'],
            data['Sex'],
            data['HighChol'],
            data['Smoker'],
            data['PhysActivity']
        ]).reshape(1, -1)

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        return jsonify({
            'diabetic': int(prediction),
            'probability': float(probability)
        })

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500


@app.route('/', methods=['GET'])
def index():
    return 'Health.ai API running!'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
