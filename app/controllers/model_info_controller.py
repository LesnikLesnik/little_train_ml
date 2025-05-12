from flask import Blueprint, render_template
import os

model_info_bp = Blueprint('model_info', __name__, template_folder='../templates')


@model_info_bp.route('/model-info')
def show_model_info():
    # Данные о модели (можно вынести в конфиг или отдельный файл)
    model_info = {
        'task': {
            'goal': "Выявление случаев депрессии среди студентов на основе анкетирования",
            'application': "Анализ анкет из 14 вопросов",
            'dataset': {
                'name': "Student Depression Dataset",
                'link': "https://www.kaggle.com/datasets/hopesb/student-depression-dataset",
                'size': "28 тыс. строк",
                'features': "14 значимых признаков (числовые и категориальные)",
                'balance': "Сбалансированные данные: 60% - класс 'депрессия'"
            }
        },
        'algorithm': {
            'name': "Random Forest",
            'advantages': [
                "Оптимален для бинарной классификации",
                "Эффективнее простого дерева решений",
                "Хорошая интерпретируемость результатов"
            ]
        },
        'metrics': {
            'confusion_matrix': [[1841, 472], [400, 2867]],
            'classification_report': {
                'precision': [0.82, 0.86],
                'recall': [0.80, 0.88],
                'f1': [0.81, 0.87],
                'support': [2313, 3267]
            },
            'accuracy': 0.84,
            'roc_auc': 0.9136
        },
        'images': [
            'img.png',
            'img_1.png',
            'img_2.png',
            'img_3.png',
            'img_4.png'
        ]
    }

    return render_template('model_info.html', model_info=model_info)