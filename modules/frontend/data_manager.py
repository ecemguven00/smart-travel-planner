import pandas as pd
import streamlit as st
import os

# --- FILE PATH SETTINGS  ---
# Finds the exact directory where this script is located.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Constructs the full path to the CSV file, assuming it's in the same directory.
FILE_PATH = os.path.join(SCRIPT_DIR, "Worldwide_Travel_Cities_WithAirport_Precipitation.csv")

# --- CONSTANTS ---
ACTIVITY_LABELS = {
    'culture': 'Culture & History', 'adventure': 'Adventure & Action',
    'nature': 'Nature & Scenery', 'beaches': 'Beaches & Sea',
    'nightlife': 'Nightlife', 'cuisine': 'Cuisine & Gastronomy',
    'wellness': 'Wellness & Health', 'urban': 'Urban Life',
    'seclusion': 'Seclusion & Peace'
}

TRIP_DURATION_OPTIONS = {
    'Short Trip (<3 Days)': 'short_trip',
    'Weekend': 'weekend',
    'One Week': 'one_week',
    'Long Trip (>1 Week)': 'long_trip',
    'Day Trip': 'day_trip'
}

SPECIAL_FILTERS = {
    'Alcohol-free': 'Alcohol-free',
    'Halal-friendly': 'Halal-friendly',
    'Safe': 'Safe',
    'family_friendly': 'Family Friendly',
    'airport_closeness': 'Close to Airport'
}

ACTIVITY_COLS = list(ACTIVITY_LABELS.keys())

@st.cache_data
def load_data():
    try:
        # We use the FILE_PATH variable to avoid path errors
        df = pd.read_csv(
            FILE_PATH,
            sep=',',
            quotechar='"',
            escapechar='\\',
            doublequote=True,
            engine='python'
        )
    except FileNotFoundError:
        st.error(f"CSV file not found at: {FILE_PATH}. Please make sure the file is in the same directory as this script.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Critical CSV Read Error: {e}")
        return pd.DataFrame()

    # --- Data Cleaning ---

    # 1. Clean Numeric Columns
    numeric_cols_to_clean = ['budget_numeric', 'avg_temp_summer', 'avg_temp_winter', 'latitude', 'longitude',
                             'distance_to_airport_km']
    for col in numeric_cols_to_clean:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val if pd.notna(mean_val) else 0)

    # 2. Boolean/Trip columns to int
    bool_cols = ['Alcohol-free', 'Halal-friendly', 'Safe', 'family_friendly', 'airport_closeness', 'short_trip',
                 'weekend', 'long_trip', 'one_week', 'day_trip']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].replace({True: 1, False: 0, 'True': 1, 'False': 0}).fillna(0).astype(int)

    # 3. Activity columns to numeric
    for col in ACTIVITY_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).round(0).astype(int)
            # Scaling 0-5 scores to 0-100 for better visualization
            if df[col].max() <= 10:
                df[col] = df[col] * 20

    # 4. Capitalize Region Names (Fix for lowercase regions)
    if 'region' in df.columns:
        df['region'] = df['region'].astype(str).str.title()

    # Handle JSON avg_temp_monthly separately
    if 'avg_temp_monthly' in df.columns:
        def safe_extract(x):
            if isinstance(x, (int, float)): return x
            return 20.0 # Default fallback

        df['avg_temp_monthly'] = df['avg_temp_monthly'].apply(safe_extract)

    return df.dropna(subset=['city', 'country'])
