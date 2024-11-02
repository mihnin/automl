import streamlit as st
import pandas as pd
from autogluon.tabular import TabularPredictor
import os

# Загрузка данных
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Выполнение прогнозирования
def perform_prediction(data, target_column):
    predictor = TabularPredictor(label=target_column).fit(data)
    predictions = predictor.predict(data)
    return predictions, predictor

# Визуализация результатов
def visualize_results(predictions, target_column):
    st.write(f"Прогнозы для столбца '{target_column}':")
    st.write(predictions)

# Сохранение результатов в Excel
def save_results_to_excel(predictions, file_path):
    predictions.to_excel(file_path, index=False)

# Основное приложение Streamlit
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

        # Выбор времени прогнозирования
        prediction_time = st.slider("Выберите время прогнозирования (секунды)", 1, 600, 60)

        if st.button("Выполнить прогнозирование"):
            with st.spinner("Прогнозирование..."):
                predictions, predictor = perform_prediction(data, target_column)
                visualize_results(predictions, target_column)

                # Таблица с результатами
                st.write("Таблица с результатами:")
                st.write(predictions)

                # Графики
                st.write("Графики:")
                st.line_chart(predictions)

                # Сохранение результатов в Excel
                if st.button("Сохранить результаты в Excel"):
                    save_results_to_excel(predictions, "predictions.xlsx")
                    st.success("Результаты сохранены в predictions.xlsx")

                # Таблица с результатами, где видно, какая модель лучше справилась
                st.write("Таблица с результатами, где видно, какая модель лучше справилась:")
                leaderboard = predictor.leaderboard(data)
                st.write(leaderboard)

if __name__ == "__main__":
    main()
