# AutoGluon App

Это приложение использует AutoGluon для прогнозирования целевого столбца в загруженных данных. Приложение разработано с использованием Streamlit для создания интерактивного интерфейса.

## Установка

1. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

2. Запустите приложение:
   ```bash
   streamlit run app/main.py
   ```

## Структура проекта

- `data/`: Каталог для хранения данных.
- `models/`: Каталог для хранения моделей.
- `utils/`: Каталог для хранения вспомогательных скриптов.
- `main.py`: Основной скрипт для загрузки данных, выполнения прогнозирования и визуализации результатов.
- `requirements.txt`: Файл с зависимостями проекта.
- `README.md`: Файл с описанием проекта.

## Использование

1. Поместите файл данных в каталог `data/`.
2. Запустите приложение с помощью команды `streamlit run app/main.py`.
3. В интерфейсе Streamlit загрузите файл данных, выберите целевой столбец и время прогнозирования.
4. Нажмите кнопку "Выполнить прогнозирование" для получения результатов.
5. Результаты будут отображены в виде таблицы и графика.
6. Вы можете сохранить результаты в Excel, нажав кнопку "Сохранить результаты в Excel".

## Пример

```python
# Путь к файлу данных
file_path = "data/dataset.csv"

# Целевой столбец для прогнозирования
target_column = "target"
```

## Лицензия

Этот проект лицензирован под MIT License.
