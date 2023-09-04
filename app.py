import shap
from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)
data = pd.read_csv("data.csv")
data = data.drop(["Unnamed: 0"], axis=1)

with open("model_knc.pkl", "rb") as f:
    model_knc = pickle.load(f)

with open("best_threshold.pickle", "rb") as f:
    thres = pickle.load(f)

with open("model.pkl", "rb") as f:
    best_model = pickle.load(f)

with open("shap_val.pickle", "rb") as f:
    shap_values = pickle.load(f)

train_x = data.iloc[5:21]
train_x = train_x.drop(["SK_ID_CURR"], axis=1)
explainer = shap.KernelExplainer(model_knc.predict_proba, train_x)


def predict(index, data, best_model, thres):
    data_id = data.loc[[index]]
    proba = float(best_model.predict_proba(data_id)[:, 1])
    rep = "Rejetée" if proba > thres else 'Acceptée'
    return rep, round(proba, 2)


@app.route('/', methods=['GET', 'POST'])
def home():
    return "Home application"


@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    try:
        index = request.get_json()
        index = int(index["value"])
        data_index = data.iloc[[index]]
        rep, proba = predict(index, data_index, best_model, thres)
        d = {'rep': rep, 'proba': proba}
        return jsonify(d)
    except Exception as e:
        return jsonify({"erreur": str(e)})


@app.route('/shap_plot', methods=['POST'])
def serve_shap_plot():
    index = request.get_json()
    elt_index = int(index["value"])
    X = data.drop(['SK_ID_CURR'], axis=1)
    image = shap.force_plot(explainer.expected_value[1], shap_values[1][elt_index], X.iloc[elt_index, :])
    shap.save_html("image.html", image)

    with open('image.html', 'r', encoding='utf-8') as file:
        html_content = file.read()

    return jsonify({"key": html_content})


if __name__ == "__main__":
    app.run(debug=True, port="5000", host="0.0.0.0")
