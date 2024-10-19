import streamlit as st
import pandas as pd
import numpy as np
import io
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go

# Функции из models.py
def process_data(df, target_column, date_column, category_column=None):
    """
    Обработка данных для прогнозирования
    """
    df = df.set_index(date_column)
    df = df.sort_index()
    
    if category_column:
        df = pd.get_dummies(df, columns=[category_column])
    
    return df[[target_column]]

def train_model(data, model_type):
    """
    Обучение модели временных рядов
    """
    if model_type == 'arima':
        model = ARIMA(data, order=(1,1,1))
    elif model_type == 'sarima':
        model = SARIMAX(data, order=(1,1,1), seasonal_order=(1,1,1,12))
    elif model_type == 'ets':
        model = ExponentialSmoothing(data, seasonal_periods=12, trend='add', seasonal='add')
    elif model_type == 'naive':
        return None  # Наивная модель не требует обучения
    else:
        raise ValueError("Неподдерживаемый тип модели")
    
    return model.fit()

def make_forecast(model, test_data):
    """
    Выполнение прогноза
    """
    if model is None:  # Наивная модель
        return test_data.shift(1)
    
    forecast = model.forecast(steps=len(test_data))
    return forecast

def evaluate_model(actual, predicted, metric):
    """
    Оценка качества модели
    """
    if metric == 'mae':
        return mean_absolute_error(actual, predicted)
    elif metric == 'mse':
        return mean_squared_error(actual, predicted)
    elif metric == 'rmse':
        return np.sqrt(mean_squared_error(actual, predicted))
    else:
        raise ValueError("Неподдерживаемая метрика")

# Настройка страницы Streamlit
st.set_page_config(page_title="Прогнозирование временных рядов", layout="wide")
st.title("Прогнозирование временных рядов")

# Загрузка файла
uploaded_file = st.file_uploader("Выберите CSV файл", type="csv")

if uploaded_file is not None:
    # Чтение CSV-файла
    df = pd.read_csv(uploaded_file)
    st.write("Предварительный просмотр данных:")
    st.write(df.head())

    # Выбор параметров
    columns = df.columns.tolist()
    target_column = st.selectbox("Выберите целевую колонку", columns)
    date_column = st.selectbox("Выберите колонку с датами", columns)
    category_column = st.selectbox("Выберите колонку с категориями (необязательно)", ["Нет"] + columns)
    category_column = None if category_column == "Нет" else category_column

    train_percent = st.slider("Выберите процент данных для обучения", 50, 90, 80)
    model_type = st.selectbox("Выберите тип модели", ["arima", "sarima", "ets", "naive"])
    metric = st.selectbox("Выберите метрику оценки", ["mae", "mse", "rmse"])

    if st.button("Выполнить прогноз"):
        # Обработка данных
        df[date_column] = pd.to_datetime(df[date_column])
        processed_data = process_data(df, target_column, date_column, category_column)

        # Разделение на обучающую и тестовую выборки
        train_size = int(len(processed_data) * train_percent / 100)
        train_data = processed_data[:train_size]
        test_data = processed_data[train_size:]

        # Обучение модели
        model = train_model(train_data, model_type)

        # Выполнение прогноза
        forecast = make_forecast(model, test_data)

        # Оценка модели
        evaluation = evaluate_model(test_data[target_column], forecast, metric)

        # Визуализация результатов
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=processed_data.index, y=processed_data[target_column],
                                 mode='lines', name='Фактические данные'))
        fig.add_trace(go.Scatter(x=test_data.index, y=forecast,
                                 mode='lines', name='Прогноз'))
        fig.update_layout(title='Фактические данные и прогноз',
                          xaxis_title='Дата',
                          yaxis_title=target_column)
        st.plotly_chart(fig)

        st.write(f"Оценка модели ({metric.upper()}): {evaluation:.4f}")

st.write("Разработано с использованием Streamlit")
