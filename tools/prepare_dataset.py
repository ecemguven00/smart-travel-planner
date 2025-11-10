import pandas as pd
import numpy as np
import json
import ast
from sklearn.neighbors import BallTree

pd.set_option('display.max_columns', None)  # Tüm sütunları göster
pd.set_option('display.width', 200)

df_worldwide = pd.read_csv("../data/Worldwide_Travel_Cities.csv")
airports = pd.read_csv("../data/airports.csv")
countries = pd.read_csv("../data/countries.csv")
worldwide = df_worldwide.copy()

#print("=== First 5 rows ===")
#print(worldwide.head(20))

#worldwide.info()
#worldwide.describe().T.plot(kind='bar')
#worldwide.isnull().sum()
#worldwide.duplicated().sum()

#print('Columns available:', worldwide.columns.tolist())

#unique_countries = worldwide['country'].nunique()
#print("Number of unique countries:", unique_countries)

#null_countries = worldwide['country'].isnull().sum()

#print("Number of null values in Country column:", null_countries)

#print("Unique countries:\n", worldwide['country'].unique())

alcohol_free_countries = ['United Arab Emirates', 'Morocco', 'Egypt', 'Indonesia']
halal_friendly_countries = ['United Arab Emirates', 'Morocco', 'Egypt', 'Indonesia']

worldwide['Alcohol-free'] = worldwide['country'].isin(alcohol_free_countries).astype(int)
worldwide['Halal-friendly'] = worldwide['country'].isin(halal_friendly_countries).astype(int)
#print(worldwide[['city', 'country', 'Alcohol-free', 'Halal-friendly']].head(20))

safe_countries = [
    'United Arab Emirates', 'Morocco', 'Egypt', 'Indonesia', 'Japan',
    'Canada', 'Australia', 'New Zealand', 'Germany', 'Netherlands', 'Sweden'
]

worldwide['Safe'] = worldwide['country'].isin(safe_countries).astype(int)

#print(worldwide[['city', 'country', 'Safe']].head(20))
#print('Columns available:', worldwide.columns.tolist())

#print(worldwide[worldwide.isnull().any(axis=1)])

#country_counts = worldwide['country'].value_counts()

budget_mapping = {
    'Budget': 1,
    'Mid-range': 2,
    'Luxury': 3
}
worldwide['budget_numeric'] = worldwide['budget_level'].map(budget_mapping)
#print(worldwide[['budget_level', 'budget_numeric']].head(10))

#null_count = worldwide['ideal_durations'].isnull().sum()

#print("Null values in 'ideal_durations':", null_count)

#unique_values = worldwide['ideal_durations'].unique()
#print("Unique values in 'ideal_durations':", unique_values)

#print("Number of unique values:", len(unique_values))
import ast

# Tüm unique duration türlerini çıkart
all_durations = set()
for row in worldwide['ideal_durations']:
    try:
        durations = ast.literal_eval(row)
        all_durations.update(durations)
    except:
        pass

print("All unique duration types:", all_durations)
for d in all_durations:
    worldwide[d.replace(" ", "_").lower()] = worldwide['ideal_durations'].apply(lambda x: 1 if d in str(x) else 0)

# Control
#worldwide[['ideal_durations', 'short_trip', 'one_week', 'weekend', 'day_trip', 'long_trip']].head()
#print(worldwide.columns)
import json

# JSON string'i dict'e çevir
worldwide['avg_temp_parsed'] = worldwide['avg_temp_monthly'].apply(json.loads)
def seasonal_avg(temp_dict, months):
    try:
        temps = [temp_dict[str(m)]["avg"] for m in months]
        return sum(temps) / len(temps)
    except:
        return None

# Summer: June, July, August
summer_months = [6, 7, 8]
winter_months = [12, 1, 2]

worldwide['avg_temp_summer'] = worldwide['avg_temp_parsed'].apply(lambda x: seasonal_avg(x, summer_months))
worldwide['avg_temp_winter'] = worldwide['avg_temp_parsed'].apply(lambda x: seasonal_avg(x, winter_months))
#print(worldwide[['city', 'country', 'avg_temp_summer', 'avg_temp_winter']].head(10))
worldwide['family_friendly'] = (
    ((worldwide['beaches'] >= 3) | (worldwide['nature'] >= 3) | (worldwide['seclusion'] >= 3)) &
    (worldwide['adventure'] < 4) &
    (worldwide['nightlife'] < 4)
).astype(int)

# First 10 rows 
#print(worldwide[['city', 'country', 'family_friendly']].head(10))
#print(worldwide.head(5))

# Only get airports of type 'airport'
airports_filtered = airports[airports['type'].isin(['large_airport','medium_airport'])]


# Get the required columns
airports_small = airports_filtered[['name', 'latitude_deg', 'longitude_deg', 'iso_country', 'municipality']]
airports_small.rename(columns={
    'name':'airport_name',
    'latitude_deg':'lat_airport',
    'longitude_deg':'lon_airport',
    'municipality':'city'
}, inplace=True)

# 2️⃣ Create the countries_small dataset
countries_small = countries[['code','name']].copy()
countries_small = countries_small.rename(columns={'code':'iso_country','name':'country_name'})


# Merge countries
airports_small = airports_small.merge(countries_small, on='iso_country', how='left')

# Radian conversion for BallTree
# 4️⃣ Convert coordinates to radians
airports_coords = np.radians(airports_small[['lat_airport','lon_airport']].values)
cities_coords = np.radians(worldwide[['latitude', 'longitude']].values)

# 5️⃣ BallTree creation
tree = BallTree(airports_coords, metric='haversine')

# 6️⃣ Find the nearest airport for each city
distances, indices = tree.query(cities_coords, k=1)
distances_km = distances.flatten() * 6371.0

# Score function (completely clear)
def distance_to_score(distance):
    if distance < 10:
        return 5
    elif distance < 30:
        return 4
    elif distance < 60:
        return 3
    elif distance < 100:
        return 2
    else:
        return 1


# Add score, nearest airport and distance
worldwide["airport_closeness"] = [distance_to_score(d) for d in distances_km]
worldwide['nearest_airport'] = [airports_small.iloc[i]['airport_name'] for i in indices.flatten()]
worldwide["distance_to_airport_km"] = distances_km

#print(worldwide[['city', 'country', 'airport_closeness', 'nearest_airport', 'distance_to_airport_km']].head(20))
#worldwide.drop(columns=["ideal_durations"], inplace=True)

worldwide.to_csv("../data/Worldwide_Travel_Cities1.csv", index=False)