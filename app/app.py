from flask import Flask
from app.controllers.model_info_controller import model_info_bp
from app.controllers.predict_controller import predict_bp
from app.controllers.home_controller import home_bp
from app.controllers.model_loader_controller import model_loader_bp

# Глобальное хранилище модели
ml_model = {"model": None}

def create_app():
    app = Flask(__name__)
    app.config["ml_model"] = ml_model

    app.register_blueprint(model_info_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(home_bp)
    app.register_blueprint(model_loader_bp)

    return app
