from flask import Blueprint, render_template, request, redirect, url_for, current_app
import joblib
import os
from ml.model.ensure_model import MODEL_PATH

predict_bp = Blueprint("predict", __name__, url_prefix="/predict")

def get_model():
    if current_app.config["ml_model"]["model"] is None:
        if os.path.exists(MODEL_PATH):
            current_app.config["ml_model"]["model"] = joblib.load(MODEL_PATH)
    return current_app.config["ml_model"]["model"]

@predict_bp.route("/form")
def form():
    model = get_model()
    if model is None:
        return redirect(url_for("model_loader.loading_page"))
    return render_template("form.html")

@predict_bp.route("/submit", methods=["POST"])
def submit():
    model = get_model()
    if model is None:
        return redirect(url_for("model_loader.loading_page"))

    # Пример простой логики предсказания
    input_data = request.form.get("input_text", "")
    prediction = model.predict([input_data])[0]

    return render_template("result.html", prediction=prediction)
