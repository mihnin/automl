import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
