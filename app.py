from flask import Flask, request, jsonify
import pandas as pd
import mlflow
import pickle


app = Flask(__name__)


def predict(index, data, best_model, thres):
    data_id = data.loc[[index]]
    proba = float(best_model.predict_proba(data_id)[:, 1])
    rep = "Rejetée" if proba > thres else 'Acceptée'
    return rep, round(proba, 2)


@app.route('/', methods=['POST', 'GET'])
def prediction():
    index = request.get_json()
    index = int(index["value"])
    data_index = data.iloc[[index]]
    rep, proba = predict(index, data_index, best_model, thres)
    d = {'rep': rep, 'proba': proba}
    return jsonify(d)


if __name__ == "__main__":
    data = pd.read_csv("data.csv")
    model_name = "LGBMClassifier"
    model_version = 4
    best_model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{model_version}")
    with open("best_threshold.pickle", "rb") as f:
        thres = pickle.load(f)
    app.run(debug=True)
