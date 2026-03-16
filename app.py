from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# LOAD MODEL
with open('models/depression_model.pkl', 'rb') as f:
    model = pickle.load(f)

# HOME
@app.route('/')
def home():
    return render_template('index.html')

# PREDICTION PAGE
@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

# RESULT PAGE
@app.route('/result')
def result():
    return render_template('result.html')

# ABOUT PAGE
@app.route('/about')
def about():
    return render_template('about.html')

# PREDICT
@app.route('/predict', methods=['POST'])
def predict():
    gender = int(request.form['gender'])
    academic_pressure = int(request.form['academic_pressure'])
    cgpa = float(request.form['cgpa'])
    study_satisfaction = int(request.form['study_satisfaction'])
    sleep_duration = int(request.form['sleep_duration'])
    dietary_habits = int(request.form['dietary_habits'])
    degree = int(request.form['degree'])
    suicidal_thoughts = int(request.form['suicidal_thoughts'])
    work_study_hours = int(request.form['work_study_hours'])
    financial_stress = int(request.form['financial_stress'])
    family_history = int(request.form['family_history'])

    input_data = np.array([[
        gender,
        academic_pressure,
        cgpa,
        study_satisfaction,
        sleep_duration,
        dietary_habits,
        degree,
        suicidal_thoughts,
        work_study_hours,
        financial_stress,
        family_history
    ]])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        result = "Depressed"
        message = "Based on your answers you may be experiencing depression. Please seek help!"
    else:
        result = "Not Depressed"
        message = "Based on your answers you are not showing signs of depression. Stay healthy!"

    return render_template('result.html',
                         result=result,
                         message=message)

if __name__ == '__main__':
    app.run(debug=True)