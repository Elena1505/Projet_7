from app import predict
import pandas as pd
import mlflow
import pickle

# data recovery
data = pd.read_csv("data.csv")
model_name = "LGBMClassifier"
model_version = 1
best_model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{model_version}")
with open("best_threshold.pickle", "rb") as f:
    thres = pickle.load(f)

# prediction function
pred = predict(0, data, best_model, thres)
proba = pred[1]


def test_is_the_prediction_probability_sup_or_equal_to_zero():
    assert proba >= 0


def test_is_the_prediction_probability_inf_or_equal_to_zero():
    assert proba <= 1


def test_is_the_prediction_probability_a_float():
    assert isinstance(proba, float)
