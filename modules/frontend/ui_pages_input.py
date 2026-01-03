from statistics import stdev

import streamlit as st
import pandas as pd
from data_manager import ACTIVITY_LABELS, TRIP_DURATION_OPTIONS, SPECIAL_FILTERS, ACTIVITY_COLS
from ui_utils import apply_custom_css, next_page, prev_page


#PAGE 1: DESTINATION
def show_destination_page(df):
    apply_custom_css()
    st.title("AI Travel Planner")
    st.markdown("### Find your next adventure")
    st.progress(20)

    tab1, tab2, tab3 = st.tabs([" Search by City", "Explore by Region", "Search by Country"])

    with tab1:
        st.markdown(" ")
        st.subheader("I know where I want to go")
        all_cities = sorted(df['city'].unique().tolist())
        selected_city = st.selectbox("Select Destination:", options=["Start typing..."] + all_cities,
                                     label_visibility="collapsed")
        st.markdown(" ")
        if st.button("Continue with City ➔", width="stretch", type="primary",
                     disabled=(selected_city == "Start typing..."), key="btn_city_tab"):
            st.session_state.selections['target_city'] = selected_city
            st.session_state.selections['target_region'] = None
            st.session_state.selections['target_country'] = None
            next_page()

    with tab2:
        st.markdown(" ")
        st.subheader("I'm flexible, show me a region")
        regions = sorted(df['region'].unique().tolist())
        selected_region = st.radio("Select a Region:", regions, index=None, horizontal=False)
        st.markdown(" ")
        if st.button("Explore Region ➔", width="stretch", type="primary", disabled=(selected_region is None),
                     key="btn_region_tab"):
            st.session_state.selections['target_region'] = selected_region
            st.session_state.selections['target_city'] = None
            st.session_state.selections['target_country'] = None
            next_page()

    with tab3:
        st.markdown(" ")
        st.subheader("I want to explore a specific country")
        all_countries = sorted(df['country'].unique().tolist())
        selected_country = st.selectbox("Select Country:", options=["Start typing..."] + all_countries,
                                        label_visibility="collapsed")
        st.markdown(" ")
        if st.button("Explore Country ➔", width="stretch", type="primary",
                     disabled=(selected_country == "Start typing..."), key="btn_country_tab"):
            st.session_state.selections['target_country'] = selected_country
            st.session_state.selections['target_city'] = None
            st.session_state.selections['target_region'] = None
            next_page()

    st.markdown("---")
    st.markdown("#### Trending Now")

    if 'random_cities' not in st.session_state:

        specific_cities = ['Dubai', 'Milan']
        city_df = pd.concat([df[df['city'] == c].iloc[:1] for c in specific_cities])

        random_pool = df[~df['city'].isin(specific_cities)]
        random_city = random_pool.sample(1)


        st.session_state.random_cities = pd.concat([city_df, random_city])

    cols = st.columns(4)


    with cols[0]:
        with st.container(border=True):
            st.markdown("**Turkey**")
            st.caption("Explore All")
            if st.button("Go ➔", key="btn_turkey_static", use_container_width=True):
                st.session_state.selections['target_country'] = 'Turkey'
                st.session_state.selections['target_city'] = None
                st.session_state.selections['target_region'] = None
                next_page()



    for i, (_, row) in enumerate(st.session_state.random_cities.iterrows()):
        with cols[i + 1]:
            with st.container(border=True):
                st.markdown(f"**{row['city']}**")
                st.caption(f"{row['country']}")

                if st.button(f"Go ➔", key=f"btn_go_{row.name}", use_container_width=True):
                    st.session_state.selections['target_city'] = row['city']
                    st.session_state.selections['target_country'] = row['country']
                    st.session_state.selections['target_region'] = None
                    next_page()

#PAGE 2: BUDGET
def show_budget_page(df):
    apply_custom_css()
    st.title("Budget Planning")
    target = (st.session_state.selections.get('target_city') or
              st.session_state.selections.get('target_region') or
              st.session_state.selections.get('target_country'))
    st.caption(f"Planning for: **{target}**")
    st.progress(40)

    st.subheader("What is your estimated budget?")
    selected_budget = st.session_state.selections.get('budget_level', 'Mid-range')
    b_col1, b_col2, b_col3 = st.columns(3)

    with b_col1:
        if st.button("Economy", width="stretch", type="primary" if selected_budget == 'Budget' else "secondary"):
            st.session_state.selections['budget_level'] = 'Economy'
    with b_col2:
        if st.button("Mid-range", width="stretch",
                     type="primary" if selected_budget == 'Mid-range' else "secondary"):
            st.session_state.selections['budget_level'] = 'Mid-range'
    with b_col3:
        if st.button("Luxury", width="stretch", type="primary" if selected_budget == 'Luxury' else "secondary"):
            st.session_state.selections['budget_level'] = 'Luxury'

    st.markdown(
        f"<div style='text-align: center; margin-top: 10px; color: white;'>Selected: {st.session_state.selections.get('budget_level', 'None')}</div>",
        unsafe_allow_html=True)
    st.markdown("---")
    c1, c2 = st.columns([1, 2])
    if c1.button("⬅Back", width="stretch"): prev_page()
    if c2.button("Next Step: Duration ➔", width="stretch", type="primary"): next_page()


#PAGE 3: DURATION
def show_duration_page(df):
    apply_custom_css()
    st.title("Travel Duration")
    st.progress(60)
    st.subheader("How long do you plan to stay?")
    sel_dur = st.session_state.selections.get('duration_label', 'Weekend')
    d_cols = st.columns(2)
    for i, option in enumerate(list(TRIP_DURATION_OPTIONS.keys())):
        with d_cols[i % 2]:
            if st.button(option, width="stretch", key=f"dur_{i}", type="primary" if sel_dur == option else "secondary"):
                st.session_state.selections['duration_label'] = option
                st.session_state.selections['duration_col'] = TRIP_DURATION_OPTIONS[option]

    st.markdown(
        f"""
            <div style='text-align: center; margin-top: 20px; color: white; font-weight: 500;'>
                Selected Duration: {st.session_state.selections.get('duration_label', 'Not Selected')}
            </div>
            """,
        unsafe_allow_html=True
    )
    st.markdown("---")
    c1, c2 = st.columns([1, 2])

    if c1.button("⬅Back", width="stretch"): prev_page()
    if c2.button("Next Step: Activities ➔", width="stretch", type="primary"): next_page()


#PAGE 4: ACTIVITIES
def show_activities_page(df):
    apply_custom_css()
    st.title("Interests and Filters")
    st.progress(80)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("What do you enjoy?")
        st.session_state.selections['selected_activities'] = st.multiselect(
            "Activities:", options=ACTIVITY_COLS, format_func=lambda x: ACTIVITY_LABELS[x],
            default=st.session_state.selections.get('selected_activities', ['culture'])
        )

    with col2:
        st.subheader("Special Preferences")
        st.session_state.selections['special_filters'] = st.multiselect(
            "Filters:", options=list(SPECIAL_FILTERS.keys()), format_func=lambda x: SPECIAL_FILTERS[x],
            default=st.session_state.selections.get('special_filters', [])
        )
    st.markdown("---")
    c1, c2 = st.columns([1, 2])
    if c1.button("⬅Back", width="stretch"): prev_page()
    if c2.button("Show Results! ➔", width="stretch", type="primary"): next_page()