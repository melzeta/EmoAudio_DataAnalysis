from evaluation.constants import EMOTION_COLUMNS
from evaluation.utils import clamp


def _fit_univariate_linear_regression(xs: list, ys: list) -> dict:
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)

    variance = sum((value - x_mean) ** 2 for value in xs)
    if variance == 0:
        return {"slope": 0.0, "intercept": y_mean}

    covariance = sum((x_value - x_mean) * (y_value - y_mean) for x_value, y_value in zip(xs, ys))
    slope = covariance / variance
    intercept = y_mean - (slope * x_mean)
    return {"slope": slope, "intercept": intercept}


def fit_baseline_model(train_rows: list) -> dict:
    coefficients = {}
    for emotion in EMOTION_COLUMNS:
        xs = [row[f"song_{emotion}"] for row in train_rows]
        ys = [row[f"true_{emotion}"] for row in train_rows]
        coefficients[emotion] = _fit_univariate_linear_regression(xs, ys)
    return {
        "model_type": "per_emotion_univariate_linear_baseline",
        "coefficients": coefficients,
        "training_row_count": len(train_rows),
    }


def predict_rows(model: dict, test_rows: list) -> list:
    predictions = []
    for row in test_rows:
        prediction = dict(row)
        for emotion in EMOTION_COLUMNS:
            slope = model["coefficients"][emotion]["slope"]
            intercept = model["coefficients"][emotion]["intercept"]
            raw_prediction = (slope * row[f"song_{emotion}"]) + intercept
            prediction[f"pred_{emotion}"] = round(clamp(raw_prediction), 6)
            prediction[f"abs_error_{emotion}"] = round(
                abs(prediction[f"pred_{emotion}"] - row[f"true_{emotion}"]),
                6,
            )
        predictions.append(prediction)
    return predictions

