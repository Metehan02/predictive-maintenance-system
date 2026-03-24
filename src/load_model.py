import joblib


def load_model(model_name="random_forest"):
    model_paths = {
        "random_forest": "models/random_forest_model.pkl",
        "xgboost": "models/xgboost_model.pkl"
    }

    return joblib.load(model_paths[model_name])