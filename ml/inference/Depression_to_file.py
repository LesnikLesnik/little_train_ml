import numpy as np
import pandas as pd
import joblib

# Загрузка модели
components = joblib.load('model/student_depression_model.pkl')
model = components['model']
# le = components['label_encoder']  # У нас нет меток. У нас они 0 и 1 - для микросекрвиса будут лейблы да нет

# Предобработка (как при обучении) для файлов с лишними колонками и без (тестируем есть ли разница)
no_change_file = pd.read_csv('../data/Student Depression Dataset.csv').dropna(how='any')
new_data = pd.read_csv('../data/Student Depression Dataset.csv').dropna(how='any').drop(
    ['id', 'Profession', 'City'], axis=1, errors='ignore')

# Прогнозирование
predictions = model.predict(new_data)
probabilities = model.predict_proba(new_data)[:, 1]

# Обратное преобразование в исходные метки (Yes/No)
#predicted_labels = le.inverse_transform(predictions)  # Нет в изначальных данных
predicted_labels = np.where(predictions == 1, 'да', 'нет')

# Предсказание
new_data['Predicted_Depression'] = predictions
new_data['Predicted_lable'] = predicted_labels
new_data['Probabiluty'] = probabilities
no_change_file['Predicted_Depression'] = predictions
no_change_file['Predicted_lable'] = predicted_labels
no_change_file['Probabiluty'] = probabilities
# Сохранение
new_data.to_csv('new_data_with_predictions.csv', index=False)
new_data.to_csv('ishodny_with_predictions.csv', index=False)

print("Predictions:", predictions)
print("Prediction lables:", predicted_labels)
print("Probabilities:", probabilities)
