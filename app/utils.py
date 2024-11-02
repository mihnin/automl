import pandas as pd

def save_results_to_excel(predictions, test_data, target_column, file_path):
    results_table = pd.DataFrame({
        'Фактическое значение': test_data[target_column],
        'Прогнозируемое значение': predictions,
        'Абсолютное отклонение': abs(test_data[target_column] - predictions),
        'Относительное отклонение': abs((test_data[target_column] - predictions) / test_data[target_column])
    })
    results_table.to_excel(file_path, index=False)

def save_results_to_csv(predictions, test_data, target_column, file_path):
    results_table = pd.DataFrame({
        'Фактическое значение': test_data[target_column],
        'Прогнозируемое значение': predictions,
        'Абсолютное отклонение': abs(test_data[target_column] - predictions),
        'Относительное отклонение': abs((test_data[target_column] - predictions) / test_data[target_column])
    })
    results_table.to_csv(file_path, index=False)
