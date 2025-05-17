from flask import Blueprint, render_template, redirect, url_for, current_app
import os
import joblib
from ml.model.ensure_model import MODEL_PATH

home_bp = Blueprint("home", __name__)

@home_bp.route("/")
def index():

    print("MODEL PATH:", MODEL_PATH)
    print("EXISTS:", os.path.exists(MODEL_PATH))

    if not os.path.exists(MODEL_PATH):
        return redirect(url_for("model_loader.loading_page"))

    if current_app.config["ml_model"]["model"] is None:
        current_app.config["ml_model"]["model"] = joblib.load(MODEL_PATH)

    return render_template("index.html")
