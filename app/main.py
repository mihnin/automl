import streamlit as st
import pandas as pd
from autogluon.tabular import TabularPredictor
import os
import sys

# Добавляем текущую директорию в путь поиска модулей
sys.path.append(os.path.dirname(__file__))

from data_processing import load_data, split_data
from model_training import train_model, predict
from visualization import visualize_results, visualize_model_info
from utils import save_results_to_excel, save_results_to_csv

def main():
    st.title("AutoGluon App")

    # Загрузка данных
    uploaded_file = st.file_uploader("Загрузите файл данных", type=["csv"])
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.write("Данные загружены:")
        st.write(data.head())

        # Выбор целевого столбца
        target_column = st.selectbox("Выберите целевой столбец", data.columns)

        # Выбор размера тестовой выборки
        test_size = st.slider("Выберите размер тестовой выборки (%)", 0.0, 1.0, 0.2)

        # Выбор периода прогноза
        forecast_period = st.date_input("Выберите период прогноза")

        if st.button("Выполнить прогнозирование"):
            with st.spinner("Прогнозирование..."):
                train_data, test_data = split_data(data, test_size)
                predictor = train_model(train_data, target_column)
                predictions = predict(predictor, test_data)
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

if __name__ == "__main__":
    main()
