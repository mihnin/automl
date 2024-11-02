import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def split_data(data, test_size):
    train_data = data.sample(frac=1-test_size, random_state=42)
    test_data = data.drop(train_data.index)
    return train_data, test_data
