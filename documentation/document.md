# AutoGluon: Руководство по установке и использованию

## Установка AutoGluon на Windows

```bash
# Создание виртуальной среды с Python 3.9
conda create -n myenv python=3.9 -y
conda activate myenv

# Установка необходимых пакетов
pip install -U pip
pip install -U setuptools wheel
pip install -U uv

# Установка CPU версии PyTorch
uv pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu

# Установка AutoGluon
uv pip install autogluon

Табличные данные (TabularPredictor)
Базовое использование

from autogluon.tabular import TabularPredictor

# Обучение модели
predictor = TabularPredictor(label='target_column').fit(train_data)

# Получение предсказаний
predictions = predictor.predict(test_data)

Основные параметры настройки:
eval_metric - метрика оценки (автоматически определяется типом задачи)
presets - предустановленные конфигурации ('best_quality', 'high_quality', 'medium_quality', 'optimize_for_deployment')
auto_stack - автоматическое использование стекинга моделей
time_limit - ограничение времени обучения в секундах
Отчеты из коробки:
Лидерборд моделей
Важность признаков
Метрики качества на валидации
Анализ ошибок
Временные ряды (TimeSeriesPredictor)
Базовое использование:
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

# Создание предиктора
predictor = TimeSeriesPredictor(
    prediction_length=24,
    path="models",
    target="target",
    eval_metric="MASE"
)

# Обучение
predictor.fit(
    train_data,
    presets="medium_quality",
    time_limit=600
)

# Прогнозирование
predictions = predictor.predict(train_data)

Основные параметры настройки:
prediction_length - горизонт прогнозирования
presets - предустановленные конфигурации ('fast_training', 'medium_quality', 'high_quality', 'best_quality')
eval_metric - метрика оценки (MASE, RMSE, MAE и др.)
time_limit - ограничение времени обучения
Особенности работы с временными рядами:
Поддержка вероятностного прогнозирования
Работа с ковариатами (known_covariates и past_covariates)
Обработка пропущенных значений
Поддержка нерегулярных временных рядов
Отчеты из коробки:
Вероятностные прогнозы (квантили)
Визуализация прогнозов
Метрики качества на валидации
Сравнение различных моделей
Ограничения:
Требуется Python версии 3.8, 3.9, 3.10 или 3.11
Для временных рядов все ряды должны иметь длину ≥ 3
Для обучения требуются ряды длиной ≥ 2 * prediction_length + 1
Временные ряды должны быть регулярными (равномерная дискретизация)