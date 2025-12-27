import pandas as pd
import requests
import random

# --- SETTINGS ---
GOOGLE_API_KEY = "AIzaSyCcjH28Jhxfc6d4m0ro3bDVaj_wlMOY-mM"

FILTER_MAP = {
    "culture": "most popular mausoleum symbolic structure must-see museum historical sites",
    "adventure": "adventure park zipline activity area amusement park",
    "nature": "national park natural beauty canyon waterfall",
    "beaches": "famous beaches coast",
    "nightlife": "most popular nightclub bar pub live music",
    "cuisine": "most famous best known local cuisine restaurants historic diner",
    "wellness": "best spa massage hammam wellness center",
    "urban": "most popular shopping mall square business center",
    "seclusion": "quiet natural areas lakeside forest",
    "default": "most popular tourist attractions"
}

# LOAD DATASET
import os
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))

# app.py, modules/backend klasöründe olduğu için
# 2 kat yukarı çıkıp 'data' klasörüne ulaşıyoruz
csv_path = os.path.join(current_dir, "..", "..", "data", "Worldwide_Travel_Cities.csv")

try:
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        # Eğer yukarıdaki yol tutmazsa (alternatif olarak proje ana dizini için)
        df = pd.read_csv('data/Worldwide_Travel_Cities.csv')
except Exception as e:
    print(f"WARNING: CSV file could not be loaded. Error: {e}")
    # Hata durumunda uygulama çökmesin diye boş şablon oluşturuyoruz
    df = pd.DataFrame(columns=['City', 'Country', 'Latitude', 'Longitude'])


#First Page Function
def get_all_places_with_id(city_name, holiday_type="default"):
    city_row = df[df['city'].str.lower() == city_name.lower()]

    if city_row.empty:
        return {"error": f"City '{city_name}' not found in the dataset."}

    lat = city_row.iloc[0]['latitude']
    lon = city_row.iloc[0]['longitude']

    location_coords = f"{lat},{lon}"

    api_keyword = FILTER_MAP.get(holiday_type.lower(), FILTER_MAP["default"])
    search_query = f"{city_name} {api_keyword}"

    GOOGLE_PLACES_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"

    params = {
        "query": search_query,
        "location": location_coords,
        "radius": 50000,
        "key": GOOGLE_API_KEY,
        "language": "en",
    }

    try:
        response = requests.get(GOOGLE_PLACES_URL, params=params)

        if response.status_code != 200:
            return {"error": f"API Error ({response.status_code}): Could not connect to Google."}

        places_data = response.json()
        results_list = places_data.get('results', [])

        if not results_list:
            return {"error": f"No place found for query '{search_query}'."}

        results_list.sort(key=lambda p: (p.get('rating', 0), p.get('user_ratings_total', 0)), reverse=True)

        valid_places_with_id = []
        top_results = results_list[:20]

        for p in top_results:
            if 'name' not in p or not p['name'] or 'place_id' not in p or 'geometry' not in p:
                continue

            valid_places_with_id.append({
                "name": p['name'].strip(),
                "place_id": p['place_id'],
                "lat": p['geometry']['location']['lat'],
                "lon": p['geometry']['location']['lng']
            })

        if not valid_places_with_id:
            return {"error": "No valid tourist destinations remaining after filtering."}

        count = min(len(valid_places_with_id), 6)
        random_selection = random.sample(valid_places_with_id, count)

        return {"places": random_selection}

    except Exception as e:
        return {"error": f"A general error occurred during the request: {str(e)}"}


#Second Page Function
def get_place_details(place_id):

    DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"

    params = {
        "place_id": place_id,
        "fields": "name,formatted_address,rating,user_ratings_total,opening_hours,photo,url,geometry,price_level,reviews,website",
        "key": GOOGLE_API_KEY,
        "language": "en",
    }

    try:
        response = requests.get(DETAILS_URL, params=params)

        if response.status_code != 200:
            return {"error": f"Detail API Error ({response.status_code}): Connection failed."}

        details_data = response.json()
        result = details_data.get('result', {})

        if not result:
            return {"error": "Details for the selected place were not found."}

        price_level_num = result.get('price_level')
        price_symbol = 'Unknown Cost'
        if isinstance(price_level_num, int):
            price_symbol = '$' * price_level_num

        # PHOTO REFERENCES:
        photo_refs = []
        if 'photos' in result:
            for p in result['photos'][:3]:
                photo_refs.append(p.get('photo_reference'))

        # REVIEWS:
        review_texts = []
        if 'reviews' in result:
            for review in result['reviews'][:3]:
                review_texts.append(review.get('text'))

        # OPEN STATUS: Check if currently open
        is_open = result.get('opening_hours', {}).get('open_now')

        details = {
            "name": result.get('name'),
            "address": result.get('formatted_address'),
            "rating": result.get('rating', 'N/A'),
            "total_ratings": result.get('user_ratings_total', 0),
            "url": result.get('url', 'N/A'),
            "hours": result.get('opening_hours', {}).get('weekday_text', ['Working hours unknown.']),
            "photo_refs": photo_refs,
            "review_texts": review_texts,
            "website": result.get('website'),
            "is_open": is_open,
            "lat": result.get('geometry', {}).get('location', {}).get('lat'),
            "lon": result.get('geometry', {}).get('location', {}).get('lng'),
            "price_level": price_symbol
        }

        return {"details": details}

    except Exception as e:
        return {"error": f"A general error occurred during the details request: {str(e)}"}


# Third Function: Get Photo URL
def get_place_photo_url(photo_ref):
    if not photo_ref:
        return "Photo not found."

    PHOTO_URL = "https://maps.googleapis.com/maps/api/place/photo"

    params = {
        "maxwidth": 800,
        "photoreference": photo_ref,
        "key": GOOGLE_API_KEY
    }

    request_url = requests.Request('GET', PHOTO_URL, params=params).prepare().url
    return request_url

if __name__ == "__main__":
    print("app.py is running in test mode...")