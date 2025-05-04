import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             roc_auc_score,
                             roc_curve)
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.calibration import CalibrationDisplay


# Функция вывода уникальных значений (для фронта)
def print_unique_values(columns):
    unique_vals = df[columns].unique()
    print(f"Колонка: {columns}")
    print(f"Уникальные значения ({len(unique_vals)}): {', '.join(map(str, sorted(unique_vals)))}")
    print("-" * 80)


# Загрузка данных
df = pd.read_csv('Student Depression Dataset.csv')  # всего 28к строк 11.5 (нет)/ 17.5 (да)
df = df.dropna(how='any')  # удалить все строки где есть пустые (таких всего 3)

# Удаление бесполезных столбцов
df = df.drop(['id', 'Profession', 'City'], axis=1)  # id - бесполезен, profrssion - везде студент, city не интересен

# df = df.iloc[:100] # ставить первые строки для тестов
# print(df[df['Depression']==0].count()) # сколько записей какого класса

# Вывод уникальных значений
printer = 1
if printer == 1:
    for column in df.columns:
        print_unique_values(column)

# Разделение на признаки и целевую переменную
X = df.drop('Depression', axis=1)
y = df['Depression']

# Преобразование Sleep Duration и Dietary Habits в порядковую переменную
sleep_order = ['Less than 5 hours', '5-6 hours', '6-7 hours', '7-8 hours', 'More than 8 hours', 'Others']
dietary_order = ["Unhealthy", "Moderate", "Healthy", "Others"]

# Задание порядка для всех порядковых признаков
ordinal_features = {'Sleep Duration': sleep_order,
                    'Dietary Habits': dietary_order,
                    'Financial Stress': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                    'Academic Pressure': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                    'Work Pressure': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                    'Study Satisfaction': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                    'Job Satisfaction': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]}


# Списки признаков
categorical_features = ['Degree', 'Gender', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']

ordinal_features_list = list(ordinal_features.keys())
numerical_features = ['Age', 'CGPA', 'Work/Study Hours']
# Cumulative Grade Point Average от 0 до 10

# Создание преобразователя
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
                  ('ord', OrdinalEncoder(categories=[ordinal_features[i] for i in ordinal_features_list]),
                   ordinal_features_list),
                  ('num', 'passthrough', numerical_features)])

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
# stratify нужен для равномерности распределения по классам в зависимости от размеров выборки,
# если размеры тестовой и тренеровочной выборки равны то в них попадет примерно
# одинаковое число строк каждого гласса

# Нужно чтобы потом препроцессор корректно работал в предсказании
preprocessor.fit(X_train)

# Балансировка классов с помощью SMOTE
# https://habr.com/ru/companies/otus/articles/782668/
smote = SMOTE(
    sampling_strategy="auto",  # Стратегия выборки. 'auto' - увеличение меньшего класса до размера большинственного.
    random_state=42,   # Зерно для генератора случайных чисел.
    k_neighbors=5   # Количество ближайших соседей для создания синтетических примеров.
)

# Создание пайплайна
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('smote', smote),
#  ('feature_selection', SelectKBest(f_classif, k=10)),  # ограничение числа до топовых признаков - учкорение
#  ('undersample', RandomUnderSampler(sampling_strategy=0.5)),  # это строка для уменьшения статистики большего класса
    ('classifier', RandomForestClassifier(random_state=42))  # тип модели в нашем случае случайный лес
])

# Настройка гиперпараметров для случайного леса
# Будет набираться случайный набор этих парамтров для обучения
param_grid_ = {
    'classifier__n_estimators': [1200, 800, 1000],   # число деревьев
    'classifier__max_depth': [None, 15, 10],  # (максимальная глубина дерева): Значение None (деревья растут до чистых листьев)
    'classifier__min_samples_split': [2, 3, 5, 6],  # (минимальное число образцов для разделения узла): Увеличение значения → уменьшение глубины деревьев → ускорение.
    'classifier__max_features': ['log2', 'sqrt'],   # (число признаков для выбора при разделении)
    'classifier__class_weight': ['balanced', None], # (балансировка классов):Незначительно влияет на скорость.class_weight='balanced' учитывает дисбаланс классов.
    "classifier__criterion": ["gini"]
}

# Для быстрого обучения (Лучшие параметры)
param_grid = {
    'classifier__n_estimators': [1200],   # число деревьев
    'classifier__max_depth': [15],  # (максимальная глубина дерева): Значение None (деревья растут до чистых листьев)
    'classifier__min_samples_split': [6],  # (минимальное число образцов для разделения узла): Увеличение значения → уменьшение глубины деревьев → ускорение.
    'classifier__max_features': ['sqrt'],   # (число признаков для выбора при разделении)
    'classifier__class_weight': ['balanced'], # (балансировка классов):Незначительно влияет на скорость.class_weight='balanced' учитывает дисбаланс классов.
    "classifier__criterion": ["gini"]
}

# Поиск по сетке параметров
random_search = RandomizedSearchCV(
    pipeline, param_grid, n_iter=540,
    cv=7, scoring='roc_auc', n_jobs=7, verbose=2
)
# cv - на сколько частей будет делится датасет для кросс валидации
# scoring  - по какому принципу будет выбрана лучшая модель
# n_jobs - число ядер (хотя скорее потоков) для работы
# verbose - статистика в консоли 1 - самый не подробный 3 - самая подробная

random_search.fit(X_train, y_train)

# Прогнозирование и оценка
best_model = random_search.best_estimator_   # Лучшая модель
# Возвращает предсказанные классы (0 или 1) для каждого примера в тестовой выборке.
# Использует порог по умолчанию 0.5. Если вероятность класса 1 ≥ 0.5 → предсказывает 1, иначе 0
y_pred = best_model.predict(X_test)
# Возвращает вероятности принадлежности к классу 1 для каждого примера
y_proba = best_model.predict_proba(X_test)[:, 1]

print("Лучшие параметры:", random_search.best_params_)
print("\nМатрица ошибок:")
print(confusion_matrix(y_test, y_pred))
print("\nОтчет классификации:")
print(classification_report(y_test, y_pred, target_names=['No Depression', 'Depression']))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

# ROC-кривая
# https://habr.com/ru/companies/netologyru/articles/582756/
# https://deepmachinelearning.ru/docs/Machine-learning/Classifier-evaluation/ROC-curve-AUC
# https://habr.com/ru/companies/otus/articles/809147/
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_proba):.2f}')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая')
plt.legend()
plt.show()

# Визуализация важности признаков
feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
feature_importances = pd.Series(
    best_model.named_steps['classifier'].feature_importances_,
    index=feature_names
).sort_values(ascending=False)

top = 15
plt.figure(figsize=(10, 6))
# feature_importances.index = feature_importances.index.str.wrap(30)
sns.barplot(x=feature_importances[:top].values, y=feature_importances[:top].index)
plt.title(f'Топ-{top} важных признаков')
plt.yticks(fontsize=8)
plt.ylabel('Признаки', size=12)
plt.xlabel('Степень влияния', size=12)
plt.tight_layout()
plt.show()

# Матрица ошибок
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True, fmt='d', cmap='Blues',
            xticklabels=['Нет депрессии', 'Депрессия'],
            yticklabels=['Нет депрессии', 'Депрессия'])
plt.ylabel('Факт')
plt.xlabel('Предсказание')
plt.title('Матрица ошибок')
plt.show()

# Precision-Recall кривая:
PrecisionRecallDisplay.from_predictions(y_test, y_proba)
plt.show()

# Калибровка вероятностей:
CalibrationDisplay.from_predictions(y_test, y_proba)
plt.show()

# Пайплайн для предсказаний
best_pipeline = Pipeline([
    ('preprocessor', random_search.best_estimator_.named_steps['preprocessor']),
    ('classifier', random_search.best_estimator_.named_steps['classifier'])])

# Сохранение модели
joblib.dump({
    'preprocessor': best_pipeline.named_steps['preprocessor'],
    'model': best_pipeline,
    'categorical_features': categorical_features,
    'numerical_features': numerical_features,
    'feature_names': best_pipeline.named_steps['preprocessor'].get_feature_names_out(),
    'feature_columns': X_train.columns.tolist()},
    'model/student_depression_model.pkl')
