from flask import Blueprint, render_template, request, redirect, url_for, current_app
import os
import joblib
import traceback
import pandas as pd
from ml.model.ensure_model import MODEL_PATH

predict_bp = Blueprint('predict', __name__, template_folder='../templates')

def load_model():
    """Ленивая загрузка модели из файла"""
    if current_app.config["ml_model"]["model"] is None:
        if os.path.exists(MODEL_PATH):
            model_data = joblib.load(MODEL_PATH)
            current_app.config["ml_model"]["model"] = model_data['model']
            current_app.config["ml_model"]["feature_columns"] = model_data['feature_columns']

def get_model_and_features():
    load_model()
    model = current_app.config["ml_model"].get("model")
    features = current_app.config["ml_model"].get("feature_columns")
    return model, features

@predict_bp.route("/form")
def form():
    model, _ = get_model_and_features()
    if model is None:
        return redirect(url_for("model_loader.loading_page"))
    return render_template("form.html")

@predict_bp.route("/predict", methods=["POST"])
def predict():
    model, feature_columns = get_model_and_features()
    if model is None:
        return redirect(url_for("model_loader.loading_page"))

    try:
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

        return render_template("result.html", prediction=prediction, probability=probability)

    except Exception as e:
        error_message = traceback.format_exc()
        current_app.logger.error(error_message)  # лог в файл или консоль
        return render_template("error.html", error=str(e))
