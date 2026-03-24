def predict_failure(model, row, threshold=0.3):
    prob = model.predict_proba([row])[0][1]

    if prob >= threshold:
        return 1, prob
    else:
        return 0, prob