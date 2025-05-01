from flask import Flask, render_template, request
import numpy as np
import random

app = Flask(__name__)

# Заглушка модели: возвращаем вероятность увольнения в %
def fake_model_predict(features):
    return random.randint(0, 100)

@app.route("/")
def index():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form
        features = [
            float(data.get("satisfaction")) / 100,
            float(data.get("evaluation")) / 100,
            int(data.get("projects")),
            int(data.get("hours")),
            int(data.get("tenure")),
        ]
        percent = fake_model_predict(features)
        result = f"Вероятность увольнения сотрудника: {percent}%"
    except Exception as e:
        result = f"Ошибка при обработке данных: {str(e)}"

    return render_template("result.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8080)
