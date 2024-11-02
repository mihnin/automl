from autogluon.tabular import TabularPredictor

def train_model(train_data, target_column):
    predictor = TabularPredictor(label=target_column).fit(train_data)
    return predictor

def predict(predictor, test_data):
    predictions = predictor.predict(test_data)
    return predictions
