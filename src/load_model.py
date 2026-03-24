import joblib


def load_model(path="models/random_forest_model.pkl"):
    return joblib.load(path)