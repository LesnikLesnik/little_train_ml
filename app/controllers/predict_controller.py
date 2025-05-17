from flask import Blueprint, render_template, request
import pandas as pd
import joblib
import os
from ml.model.ensure_model import download_model_if_missing

predict_bp = Blueprint('predict', __name__, template_folder='../templates')

# Глобальные переменные
model_path = 'ml/model/student_depression_model.pkl'
model = None
feature_columns = None

def load_model():
    global model, feature_columns
    if model is None:
        if not os.path.exists(model_path):
            success = download_model_if_missing()
            if not success:
                raise FileNotFoundError("Не удалось загрузить модель.")

        model_data = joblib.load(model_path)
        model = model_data['model']
        feature_columns = model_data['feature_columns']

@predict_bp.route("/form")
def form():
    return render_template("form.html")

@predict_bp.route("/predict", methods=["POST"])
def predict():
    try:
        # Загружаем модель при первом вызове
        load_model()

        data = request.form.to_dict()
        input_data = pd.DataFrame([data])

        numeric_fields = [
            'Age', 'Academic Pressure', 'Work Pressure', 'CGPA',
            'Study Satisfaction', 'Job Satisfaction', 'Work/Study Hours',
            'Financial Stress'
        ]
        for field in numeric_fields:
            input_data[field] = pd.to_numeric(input_data[field])

        input_data = input_data[feature_columns]

        prediction = int(model.predict(input_data)[0])
        probability = float(model.predict_proba(input_data)[0][1])
        # custom_threshold = 0.4
        # prediction = (probability >= custom_threshold).astype(int)

        return render_template("result.html", prediction=prediction, probability=probability)

    except Exception as e:
        return render_template("error.html", error=str(e))
