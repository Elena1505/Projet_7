import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import requests
import mlflow
import shap
import pickle
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")
with open("model_knc.pkl", "rb") as f:
    model_knc = pickle.load(f)
with open("shap_val.pickle", "rb") as f:
    shap_values = pickle.load(f)


# Pie chart
def pie_chart():
    percent_sup = 100 * (df['TARGET']).sum() / df.shape[0]
    percent_inf = 100 - percent_sup
    d = {'col1': [percent_sup, percent_inf], 'col2': ['% Non-creditworthy customer', '% Creditworthy customer']}
    d = pd.DataFrame(data=d)
    fig = px.pie(d, values='col1', names='col2', color='col2',
                 color_discrete_map={'% Non-creditworthy customer': 'red', '% Creditworthy customer': 'limegreen'})
    st.plotly_chart(fig)
    st.markdown("_This pie chart shows the proportion of creditworthy customer (92.3%) and non creditworthy"
                " customer (7.75%)._")


def feature_importances_customer(index):
    df_fi = df.head(20)
    df_fi = df.drop(["Unnamed: 0", "SK_ID_CURR"], axis=1)
    test_x = df_fi.iloc[:5]
    train_x = df_fi.iloc[5:14]
    explainer = shap.KernelExplainer(model_knc.predict_proba, train_x)
    data = test_x.iloc[[index]]
    shap_values = explainer.shap_values(data)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(shap.force_plot(explainer.expected_value[0], shap_values[0],
                              feature_names=df.columns, matplotlib=True))
    st.markdown("_This force plot can provide valuable insights into how individual features contribute to the"
                " predictions of a machine learning model. he color of each feature's bar indicates the direction of"
                " its impact on the prediction. Blue indicates a negative impact, while red indicates a positive "
                "impact._")
    st.markdown("_The baseline SHAP value (represented by a vertical line) represents the average or "
                "expected output of the model across the dataset._")


def feature_importances():
    df = pd.read_csv("data.csv")
    df = df.drop(["Unnamed: 0", "TARGET", "SK_ID_CURR"], axis=1)
    fig, ax = plt.subplots(figsize=(10, 10))
    shap.summary_plot(shap_values, features=df.columns, max_display=10)
    st.pyplot(fig)
    st.markdown("_This summary plot can be a valuable way to understand the impact of different features on the"
                " predictions of a machine learning model. The features with the largest absolute SHAP values have"
                " the most significant impact on the model's predictions. If a feature has a high positive SHAP value,"
                " it is positively contributing to the prediction._")
    st.markdown("_The color coding of each feature represents the value of the feature for each data point relative to"
                " the baseline value. Positive feature values are typically shown in shades of red, while negative "
                "values are shown in shades of blue._")


# Gender pie chart
def gender_pie_chart():
    fig = px.pie(df, names=df["CODE_GENDER"], title="Gender distribution")
    st.plotly_chart(fig)
    st.markdown("_This pie chart shows the distribution of customer genders. "
                "There is 34.1% of female and 65.9% of male._")


# Age histogram
def age_histogram(id, data):
    fig = px.histogram(df, x=df["DAYS_BIRTH"] / -365, title="Customer age distribution", nbins=5, labels={'x': 'Age'})
    fig.update_layout(bargap=0.1)
    marker = str(round(data["DAYS_BIRTH"] / -365).values[0])
    fig.add_vline(x=marker)
    fig.add_trace(go.Scatter(x=[marker], y=[5], mode="lines", marker=dict(color="black"),
                             name="Age of the customer " + id))
    st.plotly_chart(fig)
    st.markdown("_This histogram shows the age distribution of customer. There are 1434 customers aged between 20 and"
                " 30 years old, 2 657 between 30 and 40, 2 552 between 40 and 50, 2 224 between 50 and 60 and 1 133 "
                "between 60 and 70._")


# Years worked histogram
def years_worked(id, data):
    fig = px.histogram(df, x=df["DAYS_EMPLOYED"] / -365, title="Years worked distribution", nbins=10,
                       labels={'x': 'Years'})
    fig.update_layout(bargap=0.1)
    marker = str(round(data["DAYS_EMPLOYED"] / -365).values[0])
    fig.add_vline(x=marker)
    fig.add_trace(go.Scatter(x=[marker], y=[5], mode="lines", marker=dict(color="black"),
                             name="Years worked by the customer " + id))
    st.plotly_chart(fig)
    st.markdown("_This histogram shows the distribution of years worked by the customer. "
                "The largest proportion is between 2.5 and 7.5 years, with 3 166 customers._")


# Income histogram
def income(id, data):
    fig = px.histogram(df, x=df["AMT_INCOME_TOTAL"], title="Income distribution", nbins=50, labels={'x': 'Income'})
    marker = str(round(data["AMT_INCOME_TOTAL"]).values[0])
    fig.add_vline(x=marker)
    fig.add_trace(go.Scatter(x=[marker], y=[5], mode="lines", marker=dict(color="black"),
                             name="Income of the customer " + id))
    st.plotly_chart(fig)
    st.markdown("_This histogram shows the distribution of customer income. "
                "The largest proportion is between 100k and 149.9k, with 2 934 customers._")


# Children histogram
def children_pie_chart(id, data):
    fig = px.pie(df, names=df["CNT_CHILDREN"], title="Children repartition")
    st.plotly_chart(fig)
    st.markdown("_This histogram shows the distribution of the number of customer children. "
                "70.1% of customers have no children, 20.1% of customers have 1 child, 8.33% 2 children,"
                " 1.36% 3 children and less than 1% of customers have between 4 and 6 children._")


# Request prediction
def request_prediction(model_uri, index):
    j_index = json.dumps({"value": index})
    print("j_index", j_index)
    headers = {"Content-Type": "application/json"}
    request = requests.post(model_uri, data=j_index, headers=headers)
    print("req", request)
    req = request.json()
    proba = req["proba"]
    proba = str(proba)
    rep = req["rep"]
    if rep == 'Acceptée':
        pred = "creditworthy."
    else:
        pred = "non creditworthy."
    return pred, proba


def validator_id(id):
    valid_id = 0
    if id == "100002" or id == "100003" or id == "100004" or id == "100006" or id == "100007":
        valid_id = 1
    return valid_id


def id_to_index(id):
    if id == "100002":
        index = 0
    if id == "100003":
        index = 1
    if id == "100004":
        index = 2
    if id == "100006":
        index = 3
    if id == "100007":
        index = 4
    return index


def main():
    st.set_page_config(page_title="Scoring credit", page_icon=":dollar:")

    st.title('Interactive dashboard')
    st.markdown("Welcome to this Interactive Dashboard! In a first part, you will find general information"
                " about customers and feature importances. In a second part you will find more specific"
                " informations about specific customer.")
    st.markdown("You can drag your cursor over the graphs to see details. Then click on the arrow to enlarge the "
                "graphs.")

    st.markdown('**Percentage of customer creditworthiness**')
    pie_chart()
    st.markdown("**Feature importances**")
    feature_importances()

    st.subheader("Choose your customer and an action:")
    id = st.text_input('Choose a customer id among : 100002, 100003, 100004, 100006, 100007')

    info_btn = st.button('Customer informations')
    predict_btn = st.button('Creditworthiness prediction')

    if info_btn:
        valid_id = validator_id(id)
        if valid_id == 1:
            data = df[df['SK_ID_CURR'] == int(id)]
            st.subheader("Main informations about the customer " + id + ":")
            st.text("Gender: " + str(data["CODE_GENDER"].values[0]))
            st.text("Age: " + str(round(data["DAYS_BIRTH"] / -365).values[0]))
            st.text("Years worked: " + str(round(data["DAYS_EMPLOYED"] / -365).values[0]))
            st.text("Income: " + str(round(data['AMT_INCOME_TOTAL'].values[0])))
            st.text("Number of child/children: " + str(round(data['CNT_CHILDREN'].values[0])))

            st.subheader("All details about the customer " + id + ":")
            st.table(data)

            st.subheader("Comparison with other customers: ")

            gender_pie_chart()
            age_histogram(id, data)
            years_worked(id, data)
            income(id, data)
            children_pie_chart(id, data)

        else:
            st.text("Please enter a valid id!")

    if predict_btn:
        valid_id = validator_id(id)
        if valid_id == 1:
            st.subheader("Creditworthiness prediction for the customer " + id + ":")
            api_uri = 'https://scoring-credit-app-e49e74730d55.herokuapp.com/prediction'
            index = id_to_index(id)
            pred, proba = request_prediction(api_uri, index)
            st.text("The score is : " + proba + ", so the customer is " + pred)

        else:
            st.text("Please enter a valid id!")

        st.subheader("Feature importances for the customer " + id + ":")
        index = id_to_index(id)
        feature_importances_customer(index)


if __name__ == '__main__':
    main()
