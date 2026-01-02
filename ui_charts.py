import altair as alt
import pandas as pd
import streamlit as st
from data_manager import ACTIVITY_LABELS


def create_city_chart(row, selected_activities):
    """Creates a bar chart for a single city's activity scores."""
    chart_data = {label: row[col] for col, label in ACTIVITY_LABELS.items() if col in row}

    # Map selection keys back to display labels
    user_selected_labels = [ACTIVITY_LABELS.get(a, a) for a in selected_activities]

    city_chart_df = pd.DataFrame(list(chart_data.items()), columns=['Activity', 'Score'])
    city_chart_df['Type'] = city_chart_df['Activity'].apply(
        lambda x: 'Selected' if x in user_selected_labels else 'Other'
    )
    city_chart_df = city_chart_df.sort_values(by=['Type', 'Score'], ascending=[False, False])

    c_chart = alt.Chart(city_chart_df).mark_bar().encode(
        x=alt.X('Score', scale=alt.Scale(domain=[0, 100])),
        y=alt.Y('Activity', sort=None),
        color=alt.Color('Type', scale=alt.Scale(domain=['Selected', 'Other'], range=['#FF4B4B', '#e0e0e0'])),
        tooltip=['Activity', 'Score']
    ).properties(height=200)

    return c_chart


def create_map(df):
    """Displays a map with city locations."""
    return st.map(df, latitude='latitude', longitude='longitude', size=20, zoom=1)