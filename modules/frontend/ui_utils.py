import streamlit as st
import unicodedata
import base64
import os

def get_base64_of_bin_file(bin_file):
    """Resmi sisteme tanıtan yardımcı fonksiyon."""
    if not os.path.exists(bin_file):
        return None
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

#CSS STYLING
def apply_custom_css():

    bin_str = get_base64_of_bin_file('modules/frontend/img/Background.png')

    bg_style = ""
    if bin_str:
        bg_style = f"""
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/png;base64,{bin_str}");
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
            color: #FFFFFF !important;
        }}
        """

    st.markdown(f"""
        <style>
        {bg_style}

       
        h1, h2, h3, p, span, label, .stMarkdown {{
            color: #FFFFFF !important;
        }}

        /* Buton:Turuncu (#EE9B00) */
        div.stButton > button:first-child {{
            border-radius: 12px;
            padding: 0.5rem 1rem;
            background-color: #EE9B00 !important;
            
            border: 2px solid #0A9396
        }}

    
        div[data-testid="stMetric"] {{
            background-color: rgba(240, 242, 246, 0.6); 
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            
        }}
        /* Multiselect  */
            span[data-baseweb="tag"] {{
                background-color: #0A9396 !important;
                color: white !important;
            }}
            
            /* Çarpı ikonu */
            span[data-baseweb="tag"] svg {{
                fill: white !important;
            }}

            /* Multiselect odaklandığındaki (hover) çerçeve rengini de uyumlu yapalım */
            div[data-baseweb="select"] > div:focus-within {{
                border-color: #0A9396 !important;
            }}
            /*  Progress Bar  */
           
            div[data-testid="stProgress"] > div > div > div > div {{
                background-color: #0A9396 !important;
            }}
                   
        
            /* Link butonu */
            div.stLinkButton > a {{
                background-color: #0A9396 !important;
                color: white !important;
                border: none !important;
                transition: 0.3s; /* Renk geçişini yumuşatır */
            }}
            
            /* Link butonu hover */
        
            div.stLinkButton > a:hover {{
                background-color: #005F73 !important; 
                border-color: #0A9396 !important;
                text-decoration: none !important;
            }}
            
            
            div.stButton > button:hover {{
                opacity: 0.8;
                border-color: #0A9396 !important;
                color: white !important;
            }}
            
           /* Exploreda ki review kutusu */
            div[data-testid="stNotification"], div[role="alert"] {{
                background-color: rgba(240, 242, 246, 0.6) !important;
                padding: 15px !important;
                border-radius: 10px !important;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
                border: none !important;
            }}
        
            
            div[data-testid="stNotification"] div, 
            div[data-testid="stNotificationContent"] p,
            div[role="alert"] p {{
                color: #005F73 !important;
                font-weight: bold !important;
            }}
        
                
        </style>
    """, unsafe_allow_html=True)


#NAVIGATION HELPERS
def next_page():
    st.session_state.page += 1
    st.rerun()

def prev_page():
    st.session_state.page -= 1
    st.rerun()

def reset_app():
    st.session_state.page = 1
    st.session_state.selections = {}
    if 'random_cities' in st.session_state:
        del st.session_state['random_cities']
    st.rerun()

#URL NORMALIZER
def normalize_for_url(text):
    text = str(text)
    replacements = {
        'ı': 'i', 'ğ': 'g', 'ü': 'u', 'ş': 's', 'ö': 'o', 'ç': 'c',
        'İ': 'i', 'Ğ': 'g', 'Ü': 'u', 'Ş': 's', 'Ö': 'o', 'Ç': 'c',
        'I': 'i'
    }
    for src, dest in replacements.items():
        text = text.replace(src, dest)
    text = text.lower()
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    return text.replace(" ", "-").replace(".", "").replace("'", "")