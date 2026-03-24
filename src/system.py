from src.predict import predict_failure
from src.recommend import recommend_maintenance


def predict_and_recommend(model, row, threshold=0.3):
    prediction, prob = predict_failure(model, row, threshold)

    if prediction == 1:
        recs = recommend_maintenance(row)
        return {
            "prediction": "Failure likely",
            "probability": prob,
            "recommendations": recs
        }
    else:
        return {
            "prediction": "No immediate risk",
            "probability": prob,
            "recommendations": ["Monitor system"]
        }