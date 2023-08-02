import streamlit as st


def main():
    st.set_page_config(page_title="Scoring credit", page_icon=":dollar:")

    st.title('Interactive dashboard')
    st.markdown("Welcome to this Interactive Dashboard! In a first part, you will find general information"
                " about customers and feature importances. In a second part you will find more specific"
                " informations about specific customer. ")


if __name__ == '__main__':
    main()
