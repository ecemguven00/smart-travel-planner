import streamlit as st
import sys
import os
import traceback
from data_manager import ACTIVITY_LABELS

# --- RECOMMENDATION MODULE IMPORT LOGIC ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from ml_nlp.recommendation_system import (
        get_personalized_recommendations,
        recommend_cities_by_preferences,
        calculate_city_similarity,
        recommend_similar_cities_from_cluster
    )
    RECOMMENDATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Recommendation modules could not be loaded. Error: {e}")
    RECOMMENDATION_AVAILABLE = False

def show_recommendations_section(df, user_selections, filtered_df):
    """
    Displays the AI Recommendation section.
    """
    if not RECOMMENDATION_AVAILABLE:
        return

    st.header("💡 Personalized Recommendations")
    st.markdown("Based on your preferences, here are some destinations you might love!")

    # Recommendation Options
    col1, col2 = st.columns([2, 1])
    with col1:
        recommendation_method = st.selectbox(
            "Recommendation Method:",
            options=['hybrid', 'preferences', 'similarity'],
            format_func=lambda x: {
                'hybrid': '🎯 Hybrid (Best Match)',
                'preferences': '⭐ By Preferences',
                'similarity': '🔍 Similar Cities'
            }.get(x, x),
            help="Hybrid combines your preferences with similarity scores for best results."
        )
    with col2:
        num_recommendations = st.slider("Number of Recommendations", 3, 15, 5)

    try:
        with st.spinner("Finding perfect destinations for you..."):
            # Exclude cities already shown in the main results
            exclude_cities = filtered_df['city'].tolist() if not filtered_df.empty else []

            # Get Recommendations
            recommendations = get_personalized_recommendations(
                df,
                user_selections,
                method=recommendation_method,
                top_n=num_recommendations
            )

            # Filter exclusions
            if exclude_cities:
                recommendations = recommendations[~recommendations['city'].isin(exclude_cities)]

            if recommendations.empty:
                st.info("💭 Try adjusting your preferences to see more recommendations!")
            else:
                # Display Recommendations
                st.markdown(f"### 🎉 Top {len(recommendations)} Recommendations")

                for idx, (_, rec) in enumerate(recommendations.iterrows(), 1):
                    with st.container(border=True):
                        rec_col1, rec_col2 = st.columns([3, 1])

                        with rec_col1:
                            st.markdown(f"#### {idx}. 🏙️ {rec['city']}, {rec['country']}")
                            st.caption(f"📍 {rec['region']}")
                            st.markdown(f"_{rec.get('short_description', 'No description available')}_")

                        with rec_col2:
                            # Show Score
                            if 'recommendation_score' in rec:
                                score = rec['recommendation_score']
                                st.metric("Match Score", f"{score:.2f}",
                                          help="Higher score = better match with your preferences")
                            elif 'similarity_score' in rec:
                                score = rec['similarity_score']
                                st.metric("Similarity", f"{score:.2f}",
                                          help="Similarity to your selected city")

                        # Metrics
                        rec_metrics = st.columns(4)
                        with rec_metrics[0]:
                            st.metric("☀️ Summer", f"{rec.get('avg_temp_summer', 0):.1f} °C")
                        with rec_metrics[1]:
                            st.metric("❄️ Winter", f"{rec.get('avg_temp_winter', 0):.1f} °C")
                        with rec_metrics[2]:
                            budget_symbol = "$" if rec.get('budget_level') == 'Budget' else "$$" if rec.get(
                                'budget_level') == 'Mid-range' else "$$$"
                            st.metric("💰 Budget", budget_symbol)
                        with rec_metrics[3]:
                            st.metric("✈️ Airport", f"{rec.get('distance_to_airport_km', 0):.1f} km")

                        # Show Activity Scores (Only for relevant user selections)
                        selected_activities = user_selections.get('selected_activities', [])
                        if selected_activities:
                            activity_scores = []
                            for act in selected_activities:
                                if act in rec:
                                    activity_scores.append(f"{ACTIVITY_LABELS.get(act, act)}: {rec[act]}")

                            if activity_scores:
                                st.markdown("**Key Activity Scores:** " + " • ".join(activity_scores[:3]))

                        # Links
                        city_query = str(rec['city']).replace(" ", "+")
                        country_query = str(rec['country']).replace(" ", "+")
                        google_flights_url = f"https://www.google.com/travel/flights?q=Flights+to+{city_query}+{country_query}"
                        booking_url = f"https://www.booking.com/searchresults.html?ss={city_query}+{country_query}"

                        link_col1, link_col2 = st.columns(2)
                        with link_col1:
                            st.link_button("✈️ Google Flights", google_flights_url, width="stretch")
                        with link_col2:
                            st.link_button("🏨 Booking.com", booking_url, width="stretch")

                        st.markdown("---")

                # Interactive Selection for Recommendation
                st.markdown("### 🎯 Want to explore one of these?")
                selected_rec = st.selectbox(
                    "Select a recommended city to see details:",
                    options=["Choose a city..."] + recommendations['city'].tolist(),
                    key="recommendation_selector"
                )

                if selected_rec and selected_rec != "Choose a city...":
                    if st.button("🔍 View This City", width="stretch"):
                        # Set as target and redirect
                        st.session_state.selections['target_city'] = selected_rec
                        st.session_state.selections['target_region'] = None
                        st.session_state.selections['target_country'] = None
                        st.rerun()

    except Exception as e:
        st.error(f"Recommendation System Error: {str(e)}")
        with st.expander("Error Details"):
            st.code(traceback.format_exc())