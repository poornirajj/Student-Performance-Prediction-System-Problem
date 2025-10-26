from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    study_hours = float(request.form['study_hours'])
    attendance = float(request.form['attendance'])
    parental_education = request.form['parental_education']
    gender = request.form['gender']

    # Encode inputs to match training data
    parental_bachelor = 1 if parental_education == 'Bachelor' else 0
    parental_master = 1 if parental_education == 'Master' else 0
    gender_male = 1 if gender == 'Male' else 0

    features = np.array([[study_hours, attendance, parental_bachelor, parental_master, gender_male]])
    prediction = model.predict(features)[0]

    return render_template('index.html', prediction_text=f'Predicted Exam Score: {prediction:.2f}')

if __name__ == "__main__":
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
