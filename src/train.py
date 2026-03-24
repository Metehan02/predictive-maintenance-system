import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib


def load_data(path="data/raw/ai4i2020.csv"):
    df = pd.read_csv(path)

    df["temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["Type"] = df["Type"].map({"L": 0, "M": 1, "H": 2})

    df = df.drop(columns=["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"])

    X = df.drop("Machine failure", axis=1)
    y = df["Machine failure"]

    return train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


def train_model():
    X_train, X_test, y_train, y_test = load_data()

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    joblib.dump(model, "models/random_forest_model.pkl")

    return model, X_test, y_test


if __name__ == "__main__":
    train_model()