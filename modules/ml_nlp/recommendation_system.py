import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def calculate_city_similarity(df, reference_city, activity_cols=None):
    """
    Bir şehre benzer şehirleri cosine similarity ile bulur.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Şehir verileri
    reference_city : str
        Referans şehir adı
    activity_cols : list, optional
        Kullanılacak aktivite sütunları
    
    Returns:
    --------
    similar_cities : pandas.DataFrame
        Benzerlik skorlarına göre sıralanmış şehirler
    """
    if activity_cols is None:
        activity_cols = ['culture', 'adventure', 'nature', 'beaches', 'nightlife',
                        'cuisine', 'wellness', 'urban', 'seclusion']
    
    # Referans şehri bul
    ref_city = df[df['city'] == reference_city]
    if ref_city.empty:
        return pd.DataFrame()
    
    ref_city = ref_city.iloc[0]
    
    # Sayısal özellikleri seç
    feature_cols = activity_cols + ['budget_numeric', 'avg_temp_summer', 'avg_temp_winter']
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    # Özellik matrisini oluştur
    features = df[feature_cols].fillna(0).values
    ref_features = ref_city[feature_cols].fillna(0).values.reshape(1, -1)
    
    # Cosine similarity hesapla
    similarities = cosine_similarity(ref_features, features)[0]
    
    # Sonuçları DataFrame'e ekle
    result_df = df.copy()
    result_df['similarity_score'] = similarities
    
    # Referans şehri hariç tut ve sırala
    result_df = result_df[result_df['city'] != reference_city]
    result_df = result_df.sort_values('similarity_score', ascending=False)
    
    return result_df


def recommend_cities_by_preferences(df, preferences, top_n=10):
    """
    Kullanıcı tercihlerine göre şehir önerir.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Şehir verileri
    preferences : dict
        Kullanıcı tercihleri:
        - selected_activities: list (önemli aktiviteler)
        - budget_level: str
        - activity_threshold: int (minimum skor)
        - special_filters: list
        - avg_temp_preference: str ('warm', 'moderate', 'cold')
        - exclude_cities: list (hariç tutulacak şehirler)
    top_n : int
        Kaç şehir önerileceği
    
    Returns:
    --------
    recommendations : pandas.DataFrame
        Önerilen şehirler ve skorları
    """
    result_df = df.copy()
    
    # Başlangıç skoru
    result_df['recommendation_score'] = 0.0
    
    # 1. Aktivite skorlarına göre puanlama
    selected_activities = preferences.get('selected_activities', [])
    if selected_activities:
        activity_cols = [col for col in selected_activities if col in df.columns]
        if activity_cols:
            # Seçilen aktivitelerin ortalaması
            result_df['activity_score'] = result_df[activity_cols].mean(axis=1)
            # Normalize et (0-1 arası)
            max_score = result_df['activity_score'].max()
            if max_score > 0:
                result_df['activity_score'] = result_df['activity_score'] / max_score
            result_df['recommendation_score'] += result_df['activity_score'] * 0.4
    
    # 2. Bütçe uyumu
    budget_level = preferences.get('budget_level')
    if budget_level:
        budget_mapping = {'Budget': 1, 'Mid-range': 2, 'Luxury': 3}
        user_budget = budget_mapping.get(budget_level, 2)
        
        # Bütçe uyumu: aynı veya bir seviye farklıysa puan ver
        result_df['budget_match'] = 0
        result_df.loc[result_df['budget_numeric'] == user_budget, 'budget_match'] = 1.0
        result_df.loc[abs(result_df['budget_numeric'] - user_budget) == 1, 'budget_match'] = 0.5
        
        result_df['recommendation_score'] += result_df['budget_match'] * 0.2
    
    # 3. Sıcaklık tercihi
    temp_preference = preferences.get('avg_temp_preference', 'moderate')
    if temp_preference in ['warm', 'moderate', 'cold']:
        if 'avg_temp_summer' in df.columns:
            if temp_preference == 'warm':
                # Yaz sıcaklığı yüksek olanlar
                result_df['temp_score'] = (result_df['avg_temp_summer'] - result_df['avg_temp_summer'].min()) / \
                                         (result_df['avg_temp_summer'].max() - result_df['avg_temp_summer'].min())
            elif temp_preference == 'cold':
                # Yaz sıcaklığı düşük olanlar (tersine)
                result_df['temp_score'] = 1 - (result_df['avg_temp_summer'] - result_df['avg_temp_summer'].min()) / \
                                          (result_df['avg_temp_summer'].max() - result_df['avg_temp_summer'].min())
            else:  # moderate
                # Orta sıcaklıklar (20-25°C civarı)
                ideal_temp = 22.5
                result_df['temp_score'] = 1 - abs(result_df['avg_temp_summer'] - ideal_temp) / 30
                result_df['temp_score'] = result_df['temp_score'].clip(0, 1)
            
            result_df['recommendation_score'] += result_df['temp_score'] * 0.15
    
    # 4. Özel filtreler
    special_filters = preferences.get('special_filters', [])
    if special_filters:
        filter_score = 0
        for filter_col in special_filters:
            if filter_col in df.columns:
                filter_score += result_df[filter_col].fillna(0)
        
        if len(special_filters) > 0:
            filter_score = filter_score / len(special_filters)
            result_df['recommendation_score'] += filter_score * 0.15
    
    # 5. Minimum aktivite skoru filtresi
    activity_threshold = preferences.get('activity_threshold', 0)
    if activity_threshold > 0 and selected_activities:
        activity_cols = [col for col in selected_activities if col in df.columns]
        if activity_cols:
            min_scores = result_df[activity_cols].min(axis=1)
            penalty = (activity_threshold - min_scores).clip(lower=0)
            result_df['recommendation_score'] -= penalty * 0.1
            result_df['recommendation_score'] = result_df['recommendation_score'].clip(lower=0)
    
    # 6. Hariç tutulacak şehirler
    exclude_cities = preferences.get('exclude_cities', [])
    if exclude_cities:
        result_df = result_df[~result_df['city'].isin(exclude_cities)]
    
    # 7. Seyahat süresi uyumu (opsiyonel)
    duration_col = preferences.get('duration_col')
    if duration_col and duration_col in df.columns:
        duration_match = result_df[duration_col].fillna(0)
        result_df['recommendation_score'] += duration_match * 0.1
    
    # Skora göre sırala
    result_df = result_df.sort_values('recommendation_score', ascending=False)
    
    # Top N şehri döndür
    return result_df.head(top_n)


def recommend_similar_cities_from_cluster(df, city_name, clustered_df=None, top_n=5):
    """
    K-means clustering sonuçlarına göre aynı kümedeki benzer şehirleri önerir.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Şehir verileri
    city_name : str
        Referans şehir
    clustered_df : pandas.DataFrame, optional
        Küme etiketleri eklenmiş DataFrame
    top_n : int
        Kaç şehir önerileceği
    
    Returns:
    --------
    recommendations : pandas.DataFrame
        Önerilen şehirler
    """
    if clustered_df is None or 'cluster' not in clustered_df.columns:
        return pd.DataFrame()
    
    # Şehrin kümesini bul
    city_cluster = clustered_df[clustered_df['city'] == city_name]
    if city_cluster.empty:
        return pd.DataFrame()
    
    cluster_id = city_cluster.iloc[0]['cluster']
    
    # Aynı kümedeki diğer şehirleri bul
    same_cluster = clustered_df[clustered_df['cluster'] == cluster_id]
    same_cluster = same_cluster[same_cluster['city'] != city_name]
    
    if same_cluster.empty:
        return pd.DataFrame()
    
    # Aktivite skorlarına göre sırala
    activity_cols = ['culture', 'adventure', 'nature', 'beaches', 'nightlife',
                    'cuisine', 'wellness', 'urban', 'seclusion']
    activity_cols = [col for col in activity_cols if col in same_cluster.columns]
    
    if activity_cols:
        same_cluster['avg_activity'] = same_cluster[activity_cols].mean(axis=1)
        same_cluster = same_cluster.sort_values('avg_activity', ascending=False)
    
    return same_cluster.head(top_n)


def get_personalized_recommendations(df, user_selections, method='hybrid', top_n=10):
    """
    Kullanıcı seçimlerine göre kişiselleştirilmiş öneriler.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Şehir verileri
    user_selections : dict
        Kullanıcı seçimleri (session_state.selections)
    method : str
        'hybrid', 'preferences', 'similarity', 'cluster'
    top_n : int
        Kaç şehir önerileceği
    
    Returns:
    --------
    recommendations : pandas.DataFrame
        Önerilen şehirler ve detayları
    """
    if method == 'preferences':
        preferences = {
            'selected_activities': user_selections.get('selected_activities', []),
            'budget_level': user_selections.get('budget_level'),
            'activity_threshold': user_selections.get('activity_threshold', 0),
            'special_filters': user_selections.get('special_filters', []),
            'duration_col': user_selections.get('duration_col'),
            'exclude_cities': [user_selections.get('target_city')] if user_selections.get('target_city') else []
        }
        return recommend_cities_by_preferences(df, preferences, top_n)
    
    elif method == 'similarity':
        target_city = user_selections.get('target_city')
        if target_city:
            selected_activities = user_selections.get('selected_activities', [])
            return calculate_city_similarity(df, target_city, selected_activities).head(top_n)
        else:
            return pd.DataFrame()
    
    elif method == 'hybrid':
        # Hem tercihlere göre hem de benzerlik skoruna göre
        preferences = {
            'selected_activities': user_selections.get('selected_activities', []),
            'budget_level': user_selections.get('budget_level'),
            'activity_threshold': user_selections.get('activity_threshold', 0),
            'special_filters': user_selections.get('special_filters', []),
            'duration_col': user_selections.get('duration_col'),
            'exclude_cities': [user_selections.get('target_city')] if user_selections.get('target_city') else []
        }
        
        # Tercihlere göre öneriler
        pref_recommendations = recommend_cities_by_preferences(df, preferences, top_n * 2)
        
        # Eğer bir şehir seçilmişse, benzerlik skorunu da ekle
        target_city = user_selections.get('target_city')
        if target_city and not pref_recommendations.empty:
            selected_activities = user_selections.get('selected_activities', [])
            similar_cities = calculate_city_similarity(df, target_city, selected_activities)
            
            # Benzerlik skorunu ekle
            pref_recommendations = pref_recommendations.merge(
                similar_cities[['city', 'similarity_score']],
                on='city',
                how='left',
                suffixes=('', '_similarity')
            )
            
            # Hybrid skor: tercih skoru + benzerlik skoru
            pref_recommendations['similarity_score'] = pref_recommendations['similarity_score'].fillna(0)
            pref_recommendations['hybrid_score'] = (
                pref_recommendations['recommendation_score'] * 0.6 +
                pref_recommendations['similarity_score'] * 0.4
            )
            
            pref_recommendations = pref_recommendations.sort_values('hybrid_score', ascending=False)
        
        return pref_recommendations.head(top_n)
    
    else:
        return pd.DataFrame()

