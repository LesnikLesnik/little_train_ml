from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Загрузка модели
model_data = joblib.load('model/student_depression_model.pkl')
model = model_data['model']
feature_columns = model_data['feature_columns']


@app.route("/")
def index():
    return render_template("form.html")


from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Загрузка модели
model_data = joblib.load('model/student_depression_model.pkl')
model = model_data['model']
feature_columns = model_data['feature_columns']


@app.route("/")
def index():
    return render_template("form.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Получаем данные из формы
        data = request.form.to_dict()

        # Преобразуем данные в DataFrame
        input_data = pd.DataFrame([data])

        # Конвертируем числовые поля
        numeric_fields = [
            'Age', 'Academic Pressure', 'Work Pressure', 'CGPA',
            'Study Satisfaction', 'Job Satisfaction', 'Work/Study Hours',
            'Financial Stress'
        ]

        for field in numeric_fields:
            input_data[field] = pd.to_numeric(input_data[field])

        # Убедимся, что колонки в правильном порядке
        input_data = input_data[feature_columns]

        # Делаем предсказание
        prediction = int(model.predict(input_data)[0])
        probability = float(model.predict_proba(input_data)[0][1])

        # Рендерим шаблон с результатами
        return render_template(
            "result.html",
            prediction=prediction,
            probability=probability
        )

    except Exception as e:
        return render_template("error.html", error=str(e))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
