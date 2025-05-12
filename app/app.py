from flask import Flask
from app.controllers.model_info_controller import model_info_bp
from app.controllers.predict_controller import predict_bp
from app.controllers.home_controller import home_bp


def create_app():
    app = Flask(__name__)

    # Регистрация контроллеров (Blueprints)
    app.register_blueprint(model_info_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(home_bp)

    return app
