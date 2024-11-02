from autogluon.tabular import TabularPredictor
from autogluon.timeseries import TimeSeriesPredictor
import time

def train_model(train_data, target_column, task_type, presets, time_limit, forecast_period, progress_bar, models):
    if task_type == 'tabular':
        predictor = TabularPredictor(label=target_column)
        predictor.fit(train_data, presets=presets, time_limit=time_limit)
    elif task_type == 'timeseries':
        # Добавляем столбец item_id с уникальными значениями
        train_data['item_id'] = range(len(train_data))
        
        predictor = TimeSeriesPredictor(
            target=target_column,
            prediction_length=forecast_period,
            eval_metric="MASE",
            freq='D'  # Предположим, что данные имеют ежедневную частоту
        )
        
        # Ограничиваем список моделей только указанными
        allowed_models = [
            "NaiveModel", "SeasonalNaiveModel", "AverageModel", "SeasonalAverageModel", 
            "ZeroModel", "ETSModel", "AutoARIMAModel", "AutoETSModel", "AutoCESModel", 
            "ThetaModel", "ADIDAModel", "CrostonClassicModel", "CrostonOptimizedModel", 
            "CrostonSBAModel", "IMAPAModel", "NPTSModel", "DeepARModel", "DLinearModel", 
            "PatchTSTModel", "SimpleFeedForwardModel", "TemporalFusionTransformerModel", 
            "TiDEModel", "WaveNetModel", "DirectTabularModel", "RecursiveTabularModel", 
            "ChronosModel"
        ]
        
        predictor.fit(train_data, presets=presets, time_limit=time_limit, hyperparameters={'models': allowed_models})
    else:
        raise ValueError("Неподдерживаемый тип задачи")
    
    # Обновление шкалы прогресса
    progress_bar.progress(50)
    
    return predictor

def predict(predictor, test_data, progress_bar):
    predictions = predictor.predict(test_data)
    
    # Обновление шкалы прогресса
    progress_bar.progress(100)
    
    return predictions
