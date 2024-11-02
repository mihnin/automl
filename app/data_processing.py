import pandas as pd

def load_data(file):
    data = pd.read_csv(file, parse_dates=['Дата'])
    return data

def split_data(data, test_size):
    train_data = data.sample(frac=1-test_size, random_state=42)
    test_data = data.drop(train_data.index)
    return train_data, test_data

def preprocess_data(data):
    # Удаление дубликатов
    data = data.drop_duplicates()
    
    # Заполнение пустых значений
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].fillna(data[col].mode()[0])
        else:
            data[col] = data[col].fillna(data[col].mean())
    
    # Нормализация числовых столбцов
    numeric_cols = data.select_dtypes(include=['number']).columns
    data[numeric_cols] = (data[numeric_cols] - data[numeric_cols].mean()) / data[numeric_cols].std()
    
    return data

def analyze_data(data, st):
    st.write("Статистика по данным:")
    st.write(data.describe(include='all'))
    
    st.write("Количество уникальных значений для категориальных столбцов:")
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        st.write(f"{col}: {data[col].nunique()} уникальных значений")
    
    st.write("Наличие пустых значений:")
    st.write(data.isnull().sum())
    
    st.write("Наличие дублей строк:")
    st.write(f"Дубликаты: {data.duplicated().sum()}")
