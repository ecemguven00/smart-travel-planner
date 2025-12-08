import streamlit as st
import unicodedata


# --- CSS STYLING ---
def apply_custom_css():
    """Injects custom CSS for buttons and metrics."""
    st.markdown("""
        <style>
        div.stButton > button:first-child {
            border-radius: 12px;
            padding: 0.5rem 1rem;
            font-weight: bold;
        }
        div[data-testid="stMetric"] {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        </style>
    """, unsafe_allow_html=True)


# --- NAVIGATION HELPERS ---
def next_page():
    """Advances to the next page in the wizard."""
    st.session_state.page += 1
    st.rerun()


def prev_page():
    """Goes back to the previous page."""
    st.session_state.page -= 1
    st.rerun()


def reset_app():
    """Resets the application state to the beginning."""
    st.session_state.page = 1
    st.session_state.selections = {}
    if 'random_cities' in st.session_state:
        del st.session_state['random_cities']
    st.rerun()


# --- URL NORMALIZER (UPDATED) ---
def normalize_for_url(text):
    """Converts text to a URL-friendly slug (e.g., 'İstanbul' -> 'istanbul')."""
    text = str(text)


    replacements = {
        'ı': 'i', 'ğ': 'g', 'ü': 'u', 'ş': 's', 'ö': 'o', 'ç': 'c',
        'İ': 'i', 'Ğ': 'g', 'Ü': 'u', 'Ş': 's', 'Ö': 'o', 'Ç': 'c',
        'I': 'i'  
    }
    for src, dest in replacements.items():
        text = text.replace(src, dest)

    # 2. Now convert to lowercase and normalize unicode
    text = text.lower()
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

    # 3. Final cleanup: replace spaces with hyphens, remove dots/apostrophes
    return text.replace(" ", "-").replace(".", "").replace("'", "")
