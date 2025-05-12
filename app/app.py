from flask import Flask, render_template, request
import pandas as pd
import joblib
from app.controllers.model_info_controller import model_info_bp

model = None
feature_columns = None

def create_app():
    global model, feature_columns

    app = Flask(__name__)
    app.register_blueprint(model_info_bp)

    # Загрузка модели
    model_data = joblib.load('ml/model/student_depression_model.pkl')
    model = model_data['model']
    feature_columns = model_data['feature_columns']

    @app.route("/")
    def index():
        return render_template("form.html")

    @app.route("/predict", methods=["POST"])
    def predict():
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
            return render_template("error.html", error=str(e))

    return app
