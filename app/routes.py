from flask import Blueprint, render_template, request, jsonify
from .models import process_data, train_model, make_forecast
import pandas as pd
import io

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/forecast', methods=['POST'])
def forecast():
    # Получение файла и параметров из формы
    file = request.files['csv-file']
    target_column = request.form['target-column']
    date_column = request.form['date-column']
    category_column = request.form.get('category-column')
    train_percent = int(request.form['train-percent'])
    model_type = request.form['model']
    metric = request.form['metric']

    # Чтение CSV-файла
    df = pd.read_csv(io.StringIO(file.stream.read().decode("UTF8")), parse_dates=[date_column])

    # Обработка данных
    processed_data = process_data(df, target_column, date_column, category_column)

    # Разделение на обучающую и тестовую выборки
    train_size = int(len(processed_data) * train_percent / 100)
    train_data = processed_data[:train_size]
    test_data = processed_data[train_size:]

    # Обучение модели
    model = train_model(train_data, model_type)

    # Выполнение прогноза
    forecast = make_forecast(model, test_data)

    # Подготовка данных для отправки на клиент
    result = {
        'dates': processed_data.index.strftime('%Y-%m-%d').tolist(),
        'actual': processed_data[target_column].tolist(),
        'predicted': forecast.tolist()
    }

    return jsonify(result)
