import streamlit as st
import pandas as pd
import altair as alt
from data_manager import ACTIVITY_LABELS, ACTIVITY_COLS
from ui_utils import apply_custom_css, prev_page, reset_app, normalize_for_url
from ui_charts import create_city_chart, create_map, create_scatter_plot, create_heatmap


# --- PAGE 5: RESULTS & VISUALIZATION ---
def show_results_page(df):
    apply_custom_css()
    st.title("🎉 Your Travel Report")
    st.progress(100)

    sels = st.session_state.selections
    target_city = sels.get('target_city')
    filtered_df = df.copy()

    # --- FILTER LOGIC ---
    if target_city:
        filtered_df = filtered_df[filtered_df['city'] == target_city]
    else:
        if sels.get('target_region'):
            filtered_df = filtered_df[filtered_df['region'] == sels.get('target_region')]
        if sels.get('target_country'):
            filtered_df = filtered_df[filtered_df['country'] == sels.get('target_country')]
        if sels.get('budget_level'):
            filtered_df = filtered_df[filtered_df['budget_level'] == sels.get('budget_level')]
        dur_col = sels.get('duration_col')
        if dur_col and dur_col in df.columns:
            filtered_df = filtered_df[filtered_df[dur_col] == 1]

        sel_acts = sels.get('selected_activities', [])
        min_score = sels.get('activity_threshold', 0)
        if sel_acts and min_score > 0:
            filtered_df = filtered_df[filtered_df[sel_acts].mean(axis=1) >= min_score]

        for spec in sels.get('special_filters', []):
            if spec in df.columns:
                filtered_df = filtered_df[filtered_df[spec] == 1]

    st.subheader(f"🔍 Found {len(filtered_df)} Destinations")

    if filtered_df.empty:
        st.warning("No cities found. Try reducing the 'Minimum Score' or changing budget.")
        c1, c2 = st.columns(2)
        if c1.button("⬅️ Change Criteria", width="stretch"): prev_page()
        if c2.button("Start Over", width="stretch"): reset_app()
    else:
        display_limit = 10
        if len(filtered_df) > display_limit:
            st.info(f"Showing top {display_limit} of {len(filtered_df)} matches.")
            display_df = filtered_df.head(display_limit)
        else:
            display_df = filtered_df

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

                st.markdown(f"_{row['short_description']}_")
                st.markdown("---")

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("☀️ Summer", f"{row['avg_temp_summer']:.1f} °C")
                with m2:
                    st.metric("❄️ Winter", f"{row['avg_temp_winter']:.1f} °C")

                cost_symbol = "$" if row['budget_level'] == 'Budget' else "$$" if row[
                                                                                      'budget_level'] == 'Mid-range' else "$$$"
                with m3:
                    st.metric("💰 Budget", cost_symbol, row['budget_level'])
                with m4:
                    st.metric("✈️ Airport", f"{row['distance_to_airport_km']:.1f} km",
                              f"*{row['nearest_airport']}*" if pd.notna(row['nearest_airport']) else None)

                # --- LINK BUTTONS ---
                st.markdown(" ")
                city_slug = normalize_for_url(row['city'])
                city_query = row['city'].replace(" ", "+")
                country_query = row['country'].replace(" ", "+")

                google_flights_url = f"https://www.google.com/travel/flights?q=Flights+to+{city_query}+{country_query}"
                booking_url = f"https://www.booking.com/searchresults.html?ss={city_query}+{country_query}"

                l1, l2 = st.columns(2)
                with l1:
                    st.link_button("✈️ Google Flights", google_flights_url, width="stretch")
                with l2:
                    st.link_button("🏨 Booking.com", booking_url, width="stretch")

                # --- CHARTS ---
                c_chart = create_city_chart(row, sels.get('selected_activities', []))
                st.altair_chart(c_chart, width="stretch")

        # --- GLOBAL VISUALIZATION ---
        if len(filtered_df) > 1:
            st.markdown("---")
            st.header("📊 Comparative Analysis")
            tab1, tab2, tab3 = st.tabs(["Map View", "Budget vs Weather", "Activity Heatmap"])

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
                    st.info("Select activities to see comparison.")

    if not filtered_df.empty:
        st.markdown("---")
        if st.button("🔄 Plan New Trip", width="stretch"): reset_app()