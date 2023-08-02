import streamlit as st
import pandas as pd
import plotly.express as px


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


def main():
    st.set_page_config(page_title="Scoring credit", page_icon=":dollar:")

    st.title('Interactive dashboard')
    st.markdown("Welcome to this Interactive Dashboard! In a first part, you will find general information"
                " about customers and feature importances. In a second part you will find more specific"
                " informations about specific customer. ")
    st.markdown('**Percentage of customer creditworthiness**')
    pie_chart()


if __name__ == '__main__':
    main()
