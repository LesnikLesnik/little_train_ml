import numpy as np
import pandas as pd
import joblib

# Загрузка модели
components = joblib.load('model/student_depression_model.pkl')
model = components['model']

# Предобработка (как при обучении) для файлов с лишними колонками и без (тестируем есть ли разница)
new_data = pd.read_csv('Student Depression Dataset.csv').dropna(how='any').drop(
    ['id', 'Profession', 'City'], axis=1, errors='ignore')

# Прогнозирование
probabilities = model.predict_proba(new_data)[:, 1]  # Вероятности класса "депрессия"
custom_threshold = 0.4  # <-- Указываем порог
predictions = (probabilities >= custom_threshold).astype(int)

predicted_labels = np.where(predictions == 1, 'да', 'нет')

# Предсказание
new_data['Predicted_Depression'] = predictions
new_data['Predicted_lable'] = predicted_labels
new_data['Probability'] = probabilities

#depression = new_data['Depression']
#new_data['Check'] = np.where(predictions == depression, 1, 0)
#print(new_data['Check'].sum())

# Сохранение
new_data.to_csv('new_data_with_predictions.csv', index=False)

print("Predictions:", predictions)
print("Prediction lables:", predicted_labels)
print("Probabilities:", probabilities)
