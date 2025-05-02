from flask import Flask, render_template, request
import pandas as pd
import joblib
from werkzeug.exceptions import BadRequest
import traceback
import sys

app = Flask(__name__)

# Загрузка модели
try:
    model_data = joblib.load("model/hr_attrition_model_total.pkl")
    preprocessor = model_data['preprocessor']
    model = model_data['model']
    label_encoder = model_data['label_encoder']
    feature_names = model_data['feature_names']
    expected_columns = model_data['feature_names_original']
    print("Модель загружена. Ожидаемые колонки:", expected_columns)
except Exception as e:
    print("Ошибка загрузки модели:", file=sys.stderr)
    traceback.print_exc()
    raise

@app.route("/")
def index():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Получаем данные из формы
        form_data = request.form.to_dict()
        print("\nПолучены данные формы:", form_data)

        # Проверяем наличие всех полей
        missing_fields = set(expected_columns) - set(form_data.keys())
        if missing_fields:
            raise ValueError(f"Отсутствуют поля: {', '.join(missing_fields)}")

        # Преобразуем типы данных
        input_dict = {}
        for col in expected_columns:
            value = form_data[col]
            # Для числовых полей преобразуем в int
            if col in ['Age', 'DailyRate', 'DistanceFromHome', 'Education',
                      'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement',
                      'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate',
                      'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
                      'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
                      'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
                      'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']:
                input_dict[col] = int(value)
            else:
                input_dict[col] = value

        # Создаем DataFrame с правильным порядком колонок
        input_df = pd.DataFrame([input_dict])[expected_columns]
        print("\nСоздан DataFrame:\n", input_df)

        # Применяем препроцессинг
        processed_data = preprocessor.transform(input_df)
        processed_df = pd.DataFrame(processed_data, columns=feature_names)
        print("\nПосле препроцессинга:\n", processed_df.head())

        # Делаем предсказание
        prediction = model.predict(processed_df)
        probability = model.predict_proba(processed_df)[0][1] * 100
        prediction_label = label_encoder.inverse_transform(prediction)[0]

        result = {
            "prediction": prediction_label,
            "probability": round(probability, 2),
            "status": "success"
        }

        return render_template("result.html", prediction=result)

    except ValueError as ve:
        error_msg = f"Ошибка в данных: {str(ve)}"
        print(error_msg, file=sys.stderr)
        return render_template("error.html", error_message=error_msg), 400
    except Exception as e:
        error_msg = f"Внутренняя ошибка сервера: {str(e)}"
        print(error_msg, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return render_template("error.html", error_message=error_msg), 500

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8080)