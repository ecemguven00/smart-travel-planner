import streamlit as st
from data_manager import load_data

# Import from the NEW modular page files
from ui_pages_input import (
    show_destination_page,
    show_budget_page,
    show_duration_page,
    show_activities_page
)
from ui_pages_results import show_results_page

# --- APP CONFIG ---
st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- SESSION STATE SETUP ---
if 'page' not in st.session_state:
    st.session_state.page = 1
if 'selections' not in st.session_state:
    st.session_state.selections = {}

# --- MAIN ROUTING ---
def main():
    # Load Data
    df = load_data()

    if df.empty:
        st.warning("Data could not be loaded. Please ensure the CSV file is in the same directory.")
        return

    # Routing Logic
    if st.session_state.page == 1:
        show_destination_page(df)
    elif st.session_state.page == 2:
        show_budget_page(df)
    elif st.session_state.page == 3:
        show_duration_page(df)
    elif st.session_state.page == 4:
        show_activities_page(df)
    elif st.session_state.page == 5:
        show_results_page(df)

if __name__ == "__main__":
    main()
