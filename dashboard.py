import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv("data.csv")


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


def validator_id(id):
    valid_id = 0
    if id == "100002" or id == "100003" or id == "100004" or id == "100006" or id == "100007":
        valid_id = 1
    return valid_id


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


def main():
    st.set_page_config(page_title="Scoring credit", page_icon=":dollar:")

    st.title('Interactive dashboard')
    st.markdown("Welcome to this Interactive Dashboard! In a first part, you will find general information"
                " about customers and feature importances. In a second part you will find more specific"
                " informations about specific customer. ")
    st.markdown('**Percentage of customer creditworthiness**')
    pie_chart()

    st.subheader("Choose your customer and an action:")
    id = st.text_input('Choose a customer id among : 100002, 100003, 100004, 100006, 100007')

    info_btn = st.button('Customer informations')

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


if __name__ == '__main__':
    main()
