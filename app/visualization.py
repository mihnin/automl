import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def visualize_results(predictions, test_data, target_column):
    st.write(f"Прогнозы для столбца '{target_column}':")
    st.write(predictions)

    # Таблица с результатами
    st.write("Таблица с результатами:")
    results_table = pd.DataFrame({
        'Фактическое значение': test_data[target_column],
        'Прогнозируемое значение': predictions,
        'Абсолютное отклонение': abs(test_data[target_column] - predictions),
        'Относительное отклонение': abs((test_data[target_column] - predictions) / test_data[target_column])
    })
    st.write(results_table)

    # Графики
    st.write("Графики:")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_data.index, y=test_data[target_column], mode='lines', name='Фактическое значение'))
    fig.add_trace(go.Scatter(x=test_data.index, y=predictions, mode='lines', name='Прогнозируемое значение'))
    st.plotly_chart(fig)

    # Тепловая карта корреляции признаков
    st.write("Тепловая карта корреляции признаков:")
    corr = test_data.select_dtypes(include=['number']).corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r')
    st.plotly_chart(fig)

def visualize_model_info(predictor, test_data):
    st.write("Информация о модели:")
    leaderboard = predictor.leaderboard(test_data)
    st.write(leaderboard)
    
    fit_summary = predictor.fit_summary()
    if 'fit_time' in fit_summary:
        st.write(f"Время обучения: {fit_summary['fit_time']} сек")
    else:
        st.write("Время обучения: Недоступно")
    
    st.write(f"Метрика: {predictor.eval_metric}")
    st.write(f"Точность: {predictor.evaluate(test_data)}")
