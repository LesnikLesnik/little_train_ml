from flask import Blueprint, render_template

model_info_bp = Blueprint('model_info', __name__, template_folder='../templates')

@model_info_bp.route('/model-info')
def show_model_info():
    return render_template('model_info.html')
