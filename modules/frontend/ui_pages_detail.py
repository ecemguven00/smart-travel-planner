
import streamlit as st
import sys
import os
import requests
import pandas as pd

#PATH CONFIGURATION
current_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.dirname(current_dir)
backend_dir = os.path.join(modules_dir, 'backend')
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

#IMPORTS
try:
    from app import get_all_places_with_id, get_place_details, get_place_photo_url
    from data_manager import ACTIVITY_LABELS
except ImportError as e:
    st.error(f"Error: Could not import app.py or data_manager.py. Please check the backend folder. Error: {e}")
    st.stop()

from ui_utils import apply_custom_css, prev_page, reset_app


def show_details_page(df):
    """
    PAGE 6: Selected City Details & Top Places to Visit (Google Places API)
    """
    apply_custom_css()

    target_city = st.session_state.selections.get('target_city')
    if not target_city:
        st.warning("No city selected.")
        if st.button("Return Home"): reset_app()
        return

    st.title(f"Exploring {target_city}")
    st.caption(
        f"{st.session_state.selections.get('target_country', 'Country')} | {st.session_state.selections.get('target_region', 'Region')}")
    st.progress(100)

    #UI STATE MANAGEMENT
    if 'current_api_suggestions' not in st.session_state:
        st.session_state.current_api_suggestions = None

    if 'selected_place_id' not in st.session_state:
        st.session_state.selected_place_id = None

    user_activities = st.session_state.selections.get('selected_activities', ['culture'])
    main_filter = user_activities[0] if user_activities else 'default'

    #MAIN VIEW
    if st.session_state.selected_place_id is None:

        display_activities = [ACTIVITY_LABELS.get(a, a).split(' ')[0] for a in user_activities]
        st.markdown(f"### Top Places to Visit in {target_city}")
        st.markdown(
            f"_Based on your primary interest: **{ACTIVITY_LABELS.get(main_filter, main_filter)}** ({', '.join(display_activities)})_")


        if st.session_state.current_api_suggestions is None or st.session_state.current_api_suggestions.get(
                'city') != target_city:
            with st.spinner(f"Finding top {main_filter.upper()} spots in {target_city}..."):
                result = get_all_places_with_id(target_city, main_filter)

            if "error" in result:
                st.error(f"Error: {result['error']}")
                st.session_state.current_api_suggestions = []
            else:
                st.session_state.current_api_suggestions = {
                    'city': target_city,
                    'places': result['places']
                }

        places = st.session_state.current_api_suggestions.get('places', [])

        if not places:
            st.warning("No specific places found matching filters. Try changing your activity selection.")
            st.markdown("---")
            if st.button("⬅Back to Results", width="stretch"):
                st.session_state.current_api_suggestions = None
                st.session_state.page = 5
                st.rerun()
            return

        # SUGGESTION LIST
        st.markdown("#### Suggested Spots:")

        # CSS to reduce vertical spacing between cards
        st.markdown("""
            <style>
                div[data-testid="stVerticalBlock"] > div > div {
                    padding-bottom: 0.5rem; 
                }
            </style>
        """, unsafe_allow_html=True)

        cols = st.columns(2)

        for idx, place in enumerate(places):
            with cols[idx % 2]:
                with st.container(border=True):
                    st.markdown(f"**{idx + 1}. {place['name']}**")
                    if st.button(f"Explore ➜", key=f"btn_{place['place_id']}", width="stretch"):
                        st.session_state.selected_place_id = place['place_id']
                        st.rerun()

        st.markdown("---")

        #MAP SECTION
        st.markdown("### Location Overview")

        map_data = pd.DataFrame(places)
        map_data.rename(columns={'lat': 'latitude', 'lon': 'longitude'}, inplace=True)

        st.map(
            map_data,
            latitude='latitude',
            longitude='longitude',
            zoom=10,
            width="stretch",
        )

        st.markdown("---")
        if st.button("⬅Back to Results", width="stretch"):
            st.session_state.current_api_suggestions = None
            st.session_state.page = 5
            st.rerun()

    # DETAIL VIEW
    else:
        place_id = st.session_state.selected_place_id

        with st.spinner("Loading place details and photo gallery..."):
            detay_result = get_place_details(place_id)

            if "error" in detay_result:
                st.error(f"Detail fetching error: {detay_result['error']}")
                if st.button("Go Back"):
                    st.session_state.selected_place_id = None
                    st.rerun()
                return

            details = detay_result['details']

            main_photo_ref = details['photo_refs'][0] if details['photo_refs'] else None
            main_photo_url = get_place_photo_url(main_photo_ref)

        if st.button("⬅Back to List"):
            st.session_state.selected_place_id = None
            st.rerun()

        # TOP INFORMATION SECTION
        st.title(details.get('name'))

        col1_img, col2_info = st.columns([1, 2])

        with col1_img:
            # Main image
            if main_photo_url and "maps.googleapis.com" in main_photo_url:
                st.image(main_photo_url, width="stretch", caption=details.get('name'))
            else:
                st.image("https://via.placeholder.com/600x400?text=No+Main+Photo", use_container_width=True,
                         caption="No Image")

            st.markdown(f"**Rating:** {details.get('rating', 'N/A')} / 5.0")
            st.caption(f"({details.get('total_ratings', 0)} votes)")

        with col2_info:
            st.subheader("General Information")
            st.markdown(f"**Address:** {details.get('address', 'Unknown')}")

            st.markdown("---")

            # COST, WEBSITE VE HOURS
            col_cost, col_hours = st.columns(2)

            with col_cost:
                st.subheader("Price Level")
                price_level_info = details.get('price_level', 'Unknown Cost')
                if price_level_info == 'Unknown Cost' or price_level_info == '':
                    st.markdown("**Entry Fee:** Free or Low Cost (Check Official Site for Details)")
                else:
                    st.markdown(f"**Entry Fee:** {price_level_info} (Price Level)")

                st.markdown("---")

                if details.get('website'):
                    st.markdown(
                        f"""
                        <div style='text-align: center; margin-top: 5px; margin-bottom: 5px;'>
                            <a href="{details['website']}" target="_blank" 
                               style="text-decoration: none; padding: 6px 12px; border: 1px solid #OA9396; border-radius: 5px; background-color: rgba(240, 242, 246, 0.6); color: #005F73;font-weight: bold; font-size: 0.9em; display: inline-block;">
                                Visit Website
                            </a>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown("<h5 style='margin-bottom: 0.5rem;'>Official Website</h5>", unsafe_allow_html=True)
                    st.info("Official website link is not available.")

            with col_hours:
                st.subheader("Working Hours")

                # OPEN/CLOSED STATUS
                is_open = details.get('is_open')
                if is_open is not None:
                    if is_open:
                        st.success("Currently OPEN")
                    else:
                        st.error("Currently CLOSED")
                else:
                    st.warning("Current status unknown.")

                hours = details.get('hours')
                if hours and isinstance(hours, list):
                    for saat in hours:
                        st.markdown(f"- {saat}")
                else:
                    st.info("Detailed working hours are not available.")


        # TRANSPORTATION SECTION
        st.markdown("---")
        st.header("Transportation and Directions")

        if details.get('url'):
            st.markdown(
                "You can use Google Maps to plan how to reach this location. Clicking the link will automatically provide direction options.")
            st.link_button("Get Directions (Google Maps)", details['url'], width="stretch")
        else:
            st.info("No map link found for this location.")

        st.markdown("---")

        #REVIEWS SECTION
        review_texts = details.get('review_texts', [])

        if review_texts:
            st.header("Visitor Reviews (Top 3)")
            for i, review in enumerate(review_texts):
                st.markdown(f"**Review #{i + 1}**")
                st.info(review)
        else:
            st.header("Visitor Reviews")
            st.info("No visitor reviews found for this location.")

        st.markdown("---")  # Separator

        #PHOTO GALLERY SECTION
        st.header("Photo Gallery")

        gallery_refs = details['photo_refs'][1:] if details['photo_refs'] else []

        if not gallery_refs:
            st.info("Additional photos are not available for this location.")
        else:
            num_cols = min(len(gallery_refs), 5)
            gal_cols = st.columns(num_cols)

            for idx, ref in enumerate(gallery_refs):
                if idx < num_cols:
                    gallery_url = get_place_photo_url(ref)
                    with gal_cols[idx]:
                        st.image(gallery_url, width="stretch")
