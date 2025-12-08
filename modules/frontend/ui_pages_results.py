import streamlit as st
import pandas as pd
import altair as alt

# --- LOCAL MODULE IMPORTS ---
from data_manager import ACTIVITY_LABELS
from ui_utils import apply_custom_css, prev_page, reset_app
from ui_charts import (
    create_city_chart, create_map, create_scatter_plot,
    create_heatmap
)

# Yeni modülleri import et
from ui_results_ml import show_ml_analysis_tab
from ui_results_recommendations import show_recommendations_section, RECOMMENDATION_AVAILABLE

# --- BACKWARD COMPATIBILITY FOR STREAMLIT ---
if not hasattr(st, 'rerun'):
    st.rerun = st.experimental_rerun

# --- PAGE 5: RESULTS & VISUALIZATION ---
def show_results_page(df):
    """
    Main function to display the results page, including filters,
    city cards, and visualization tabs.
    """
    apply_custom_css()
    st.title("🎉 Your Travel Report")
    st.progress(100)

    # Retrieve selections from session state
    sels = st.session_state.selections
    target_city = sels.get('target_city')
    filtered_df = df.copy()

    # --- FILTER LOGIC ---
    if target_city:
        filtered_df = filtered_df[filtered_df['city'] == target_city]
    else:
        # Region Filter
        if sels.get('target_region'):
            filtered_df = filtered_df[filtered_df['region'] == sels.get('target_region')]

        # Country Filter
        if sels.get('target_country'):
            filtered_df = filtered_df[filtered_df['country'] == sels.get('target_country')]

        # Budget Filter
        if sels.get('budget_level'):
            filtered_df = filtered_df[filtered_df['budget_level'] == sels.get('budget_level')]

        # Duration Filter
        dur_col = sels.get('duration_col')
        if dur_col and dur_col in df.columns:
            filtered_df = filtered_df[filtered_df[dur_col] == 1]

        # Activity Score Filter
        sel_acts = sels.get('selected_activities', [])
        min_score = sels.get('activity_threshold', 0)
        if sel_acts and min_score > 0:
            # Filter based on the average score of selected activities
            filtered_df = filtered_df[filtered_df[sel_acts].mean(axis=1) >= min_score]

        # Special Filters (Alcohol-free, Halal, etc.)
        for spec in sels.get('special_filters', []):
            if spec in df.columns:
                filtered_df = filtered_df[filtered_df[spec] == 1]

    # --- DISPLAY RESULTS ---
    st.subheader(f"🔍 Found {len(filtered_df)} Destinations")

    if filtered_df.empty:
        st.warning("No cities found. Try reducing the 'Minimum Score' or changing budget criteria.")
        c1, c2 = st.columns(2)
        if c1.button("⬅️ Change Criteria", key="btn_back_empty", width="stretch"): prev_page()
        if c2.button("Start Over", key="btn_reset_empty", width="stretch"): reset_app()
    else:
        # Limit display to avoid UI lag
        display_limit = 10
        if len(filtered_df) > display_limit:
            st.info(f"Showing top {display_limit} of {len(filtered_df)} matches.")
            display_df = filtered_df.head(display_limit)
        else:
            display_df = filtered_df

        # Render City Cards
        for index, row in display_df.reset_index().iterrows():
            with st.container(border=True):
                c_head1, c_head2 = st.columns([3, 1])
                with c_head1:
                    st.markdown(f"## 🏙️ {row['city']}, {row['country']}")
                    st.caption(f"Region: {row['region']}")

                # --- TAGS ---
                with c_head2:
                    tags = []
                    if row.get('Safe') == 1: tags.append("🛡️ Safe")
                    if row.get('family_friendly') == 1: tags.append("👨‍👩‍👧‍👦 Family")
                    if row.get('Alcohol-free') == 1: tags.append("🚫 No-Alcohol")
                    if row.get('Halal-friendly') == 1: tags.append("☪️ Halal")
                    if row.get('airport_closeness') == 1: tags.append("✈️ Near Airport")
                    if tags: st.info("  •  ".join(tags))

                st.markdown(f"_{row.get('short_description', 'No description available')}_")
                st.markdown("---")

                # Metrics Row
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("☀️ Summer", f"{row.get('avg_temp_summer', 0):.1f} °C")
                with m2:
                    st.metric("❄️ Winter", f"{row.get('avg_temp_winter', 0):.1f} °C")

                cost_symbol = "$" if row['budget_level'] == 'Budget' else "$$" if row[
                                                                                      'budget_level'] == 'Mid-range' else "$$$"
                with m3:
                    st.metric("💰 Budget", cost_symbol, row['budget_level'])
                with m4:
                    dist = row.get('distance_to_airport_km', 0)
                    airport_name = row.get('nearest_airport', '')
                    st.metric("✈️ Airport", f"{dist:.1f} km",
                              f"*{airport_name}*" if pd.notna(airport_name) else None)

                # --- LINK BUTTONS ---
                st.markdown(" ")
                city_query = str(row['city']).replace(" ", "+")
                country_query = str(row['country']).replace(" ", "+")

                google_flights_url = f"https://www.google.com/travel/flights?q=Flights+to+{city_query}+{country_query}"
                booking_url = f"https://www.booking.com/searchresults.html?ss={city_query}+{country_query}"

                l1, l2 = st.columns(2)
                with l1:
                    st.link_button("✈️ Google Flights", google_flights_url, width="stretch")
                with l2:
                    st.link_button("🏨 Booking.com", booking_url, width="stretch")

                # --- INDIVIDUAL CHART ---
                c_chart = create_city_chart(row, sels.get('selected_activities', []))
                st.altair_chart(c_chart, width="stretch")

        # --- RECOMMENDATION SECTION (AI) ---
        if RECOMMENDATION_AVAILABLE:
            st.markdown("---")
            show_recommendations_section(df, sels, filtered_df)

        # --- GLOBAL VISUALIZATION & ML TABS ---
        if len(filtered_df) > 1:
            st.markdown("---")
            st.header("📊 Comparative Analysis")
            tab1, tab2, tab3, tab4 = st.tabs(["Map View", "Budget vs Weather", "Activity Heatmap", "🔬 ML Analysis"])

            with tab1:
                create_map(filtered_df)
            with tab2:
                scatter = create_scatter_plot(filtered_df)
                st.altair_chart(scatter, width="stretch")
            with tab3:
                if sels.get('selected_activities'):
                    heatmap = create_heatmap(filtered_df, sels.get('selected_activities'))
                    st.altair_chart(heatmap, width="stretch")
                else:
                    st.info("Select activities to see the comparison.")
            with tab4:
                # Modülden çağırıyoruz
                show_ml_analysis_tab(filtered_df)

    # Footer Action
    if not filtered_df.empty:
        st.markdown("---")
        if st.button("🔄 Plan New Trip", key="btn_reset_footer", width="stretch"):
             reset_app()