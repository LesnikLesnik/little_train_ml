# Student depression Prediction

## Описание проекта

Этот проект представляет собой веб-приложение с моделью машинного обучения, которое предсказывает наличие депрессии у студентов.
Проект выполняется в учебной группе и построен по архитектуре MVC (Model-View-Controller), чтобы обеспечить разделение логики и удобство командной разработки.

<details> <summary><strong>Изображения </strong></summary>

![image_2025-05-20_18-22-44.png](app/static/images/example/start_page.png)
![image_2025-05-20_18-22-56.png](app/static/images/example/model_info.png)
![image_2025-05-20_18-23-39.png](app/static/images/example/form.png)
![image_2025-05-20_18-24-47.png](app/static/images/example/result.png)

</details>

---

## Используемый стек

- Python 3.10+
- pandas
- matplotlib
- sklearn
- imblearn
- joblib
- Flask
- HTML, CSS
- Git/GitHub

---

## Архитектура

Проект разделён по веткам:

- [`model_info`](./docs/INFO.md) — описание модели
- [`model`](./ml/training/Depression_model.py) - обучение модели
- [`dev`](./app) — разработка пользовательского интерфейса и контроллеров

---

## Запуск проекта

Поскольку работа ведется в учебной группе, где не все обладают навыками программирования, ниже предоставлены подробные вспомогательные материалы.
- Запуск проекта по [`инструкции`](docs/launch.md) 
- Инструкция по работе с [`git`](./docs/Git.md)
