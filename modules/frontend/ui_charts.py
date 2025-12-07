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


def create_scatter_plot(df):
    """Creates a scatter plot comparing Budget vs Summer Temperature."""
    scatter = alt.Chart(df).mark_circle(size=100).encode(
        x=alt.X('budget_numeric', title='Daily Budget Estimate'),
        y=alt.Y('avg_temp_summer', title='Summer Temp (°C)'),
        color='region',
        tooltip=['city', 'country', 'budget_level', 'avg_temp_summer']
    ).interactive()
    return scatter


def create_heatmap(df, selected_activities):
    """Creates a heatmap comparing cities and activity scores."""
    cols_to_keep = ['city'] + selected_activities
    melted_df = df[cols_to_keep].melt('city', var_name='activity_col', value_name='score')
    melted_df['Activity Name'] = melted_df['activity_col'].map(ACTIVITY_LABELS)

    heatmap = alt.Chart(melted_df).mark_rect().encode(
        x='Activity Name',
        y='city',
        color=alt.Color('score', title='Score (0-100)', scale=alt.Scale(scheme='reds')),
        tooltip=['city', 'Activity Name', 'score']
    ).properties(height=max(300, len(df) * 30))

    return heatmap