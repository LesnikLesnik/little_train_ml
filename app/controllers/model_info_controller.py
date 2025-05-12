from flask import Blueprint, render_template
import yaml

model_info_bp = Blueprint('model_info', __name__, template_folder='../templates')

def load_model_info(config_path="app/resources/model_info.yaml"):

    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise ValueError(f"Ошибка при разборе YAML: {e}")

    return data

@model_info_bp.route('/model-info')
def show_model_info():
    model_info = load_model_info()
    return render_template('model_info.html', model_info=model_info)
