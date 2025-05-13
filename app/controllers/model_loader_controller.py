from flask import Blueprint, render_template, redirect, url_for, current_app
import os
import joblib
from ml.model.ensure_model import download_model_if_missing, MODEL_PATH

model_loader_bp = Blueprint("model_loader", __name__)

@model_loader_bp.route("/loading")
def loading_page():
    return render_template("loading.html")

@model_loader_bp.route("/load_model", methods=["GET", "POST"])
def load_model():
    from flask import current_app
    success = download_model_if_missing()
    if success and os.path.exists(MODEL_PATH):
        current_app.config["ml_model"]["model"] = joblib.load(MODEL_PATH)
        return redirect(url_for("home.index"))

    return render_template("loading_spinner.html")

@model_loader_bp.route("/start_download", methods=["POST"])
def start_download():
    return render_template("loading_spinner.html")
