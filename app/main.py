import streamlit as st
import pandas as pd
from autogluon.tabular import TabularPredictor
import os
import sys
import time

# Добавляем текущую директорию в путь поиска модулей
sys.path.append(os.path.dirname(__file__))

from data_processing import load_data, split_data, preprocess_data, analyze_data
from model_training import train_model, predict
from visualization import visualize_results, visualize_model_info
from utils import save_results_to_excel, save_results_to_csv, save_preprocessed_data

def main():
    st.title("AutoGluon App")

    # Загрузка данных
    uploaded_file = st.file_uploader("Загрузите файл данных", type=["csv"])
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.write("Данные загружены:")
        st.write(data.head())

        # Предварительный анализ данных
        analyze_data(data, st)

        # Предобработка данных
        data = preprocess_data(data)

        # Проверка наличия столбца с временными метками
        if 'timestamp' not in data.columns:
            st.warning("Данные не содержат столбец 'timestamp'. Добавление временных меток...")
            data['timestamp'] = pd.date_range(start='1/1/2020', periods=len(data), freq='D')

        # Выбор целевого столбца
        target_column = st.selectbox("Выберите целевой столбец", data.columns)

        # Выбор типа задачи
        task_type = st.selectbox("Выберите тип задачи", ["tabular", "timeseries"])

        # Выбор предустановок
        presets = st.selectbox("Выберите предустановки", ["best_quality", "high_quality", "medium_quality", "optimize_for_deployment"])

        # Выбор размера тестовой выборки
        test_size = st.slider("Выберите размер тестовой выборки (%)", 0.0, 1.0, 0.2)

        # Выбор периода прогноза
        forecast_period = st.slider("Выберите период прогноза (в днях)", 1, 21, 5)  # Уменьшен период прогноза

        # Выбор времени для ограничения прогнозирования
        time_limit = st.slider("Выберите время для ограничения прогнозирования (в секундах)", 1, 3600, 600)

        # Выбор моделей для прогнозирования
        if task_type == "tabular":
            models = st.multiselect("Выберите модели для прогнозирования", [
                "AbstractModel", "LGBModel", "CatBoostModel", "XGBoostModel", "RFModel", 
                "XTModel", "KNNModel", "LinearModel", "TabularNeuralNetTorchModel", 
                "NNFastAiTabularModel", "VowpalWabbitModel", "MultiModalPredictorModel", 
                "TextPredictorModel", "ImagePredictorModel"
            ])
        elif task_type == "timeseries":
            models = st.multiselect("Выберите модели для прогнозирования", [
                "NaiveModel", "SeasonalNaiveModel", "AverageModel", "SeasonalAverageModel", 
                "ZeroModel", "ETSModel", "AutoARIMAModel", "AutoETSModel", "AutoCESModel", 
                "ThetaModel", "ADIDAModel", "CrostonClassicModel", "CrostonOptimizedModel", 
                "CrostonSBAModel", "IMAPAModel", "NPTSModel", "DeepARModel", "DLinearModel", 
                "PatchTSTModel", "SimpleFeedForwardModel", "TemporalFusionTransformerModel", 
                "TiDEModel", "WaveNetModel", "DirectTabularModel", "RecursiveTabularModel", 
                "ChronosModel"
            ])

        if st.button("Выполнить прогнозирование"):
            with st.spinner("Прогнозирование..."):
                train_data, test_data = split_data(data, test_size)
                progress_bar = st.progress(0)
                start_time = time.time()
                predictor = train_model(train_data, target_column, task_type, presets, time_limit=time_limit, forecast_period=forecast_period, progress_bar=progress_bar, models=models)
                predictions = predict(predictor, test_data, progress_bar=progress_bar)
                elapsed_time = time.time() - start_time
                progress_bar.progress(100)
                st.write(f"Прогнозирование завершено за {elapsed_time:.2f} секунд.")
                visualize_results(predictions, test_data, target_column)
                visualize_model_info(predictor, test_data)

                # Сохранение результатов в Excel
                if st.button("Сохранить результаты в Excel"):
                    save_results_to_excel(predictions, test_data, target_column, "predictions.xlsx")
                    st.success("Результаты сохранены в predictions.xlsx")

                # Сохранение результатов в CSV
                if st.button("Сохранить результаты в CSV"):
                    save_results_to_csv(predictions, test_data, target_column, "predictions.csv")
                    st.success("Результаты сохранены в predictions.csv")

                # Сохранение предобработанных данных
                if st.button("Сохранить предобработанные данные"):
                    save_preprocessed_data(data, "preprocessed_data.csv")
                    st.success("Предобработанные данные сохранены в preprocessed_data.csv")

if __name__ == "__main__":
    main()
