import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


def prepare_features_for_clustering(df):
    """
    Şehir verilerini clustering için hazırlar.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Şehir verilerini içeren DataFrame
    
    Returns:
    --------
    features_df : pandas.DataFrame
        Clustering için hazırlanmış özellikler
    feature_names : list
        Kullanılan özellik isimleri
    """
    # Sayısal özellikler
    numeric_features = [
        # Aktivite skorları
        'culture', 'adventure', 'nature', 'beaches', 'nightlife',
        'cuisine', 'wellness', 'urban', 'seclusion',
        # Coğrafi özellikler
        'latitude', 'longitude',
        # Sıcaklık özellikleri
        'avg_temp_summer', 'avg_temp_winter',
        # Bütçe ve mesafe
        'budget_numeric', 'distance_to_airport_km',
        # Boolean özellikler (0/1)
        'Alcohol-free', 'Halal-friendly', 'Safe', 'family_friendly',
        'airport_closeness',
        # Seyahat süresi özellikleri
        'short_trip', 'weekend', 'long_trip', 'one_week', 'day_trip'
    ]
    
    # Mevcut sütunları kontrol et
    available_features = [col for col in numeric_features if col in df.columns]
    
    # Özellikleri seç
    features_df = df[available_features].copy()
    
    # Eksik değerleri doldur
    features_df = features_df.fillna(features_df.mean())
    
    return features_df, available_features


def apply_pca(df, n_components=None, explained_variance_threshold=0.95):
    """
    PCA (Principal Component Analysis) uygular.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Şehir verilerini içeren DataFrame
    n_components : int, optional
        İstenen bileşen sayısı. None ise explained_variance_threshold kullanılır.
    explained_variance_threshold : float, default=0.95
        Açıklanan varyans eşiği (n_components None ise kullanılır)
    
    Returns:
    --------
    pca_df : pandas.DataFrame
        PCA sonuçları
    pca_model : sklearn.decomposition.PCA
        Eğitilmiş PCA modeli
    scaler : sklearn.preprocessing.StandardScaler
        Kullanılan scaler
    explained_variance_ratio : numpy.ndarray
        Her bileşenin açıklanan varyans oranı
    """
    # Özellikleri hazırla
    features_df, feature_names = prepare_features_for_clustering(df)
    
    # Standardizasyon
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df)
    
    # PCA uygula
    if n_components is None:
        # Önce tüm bileşenleri hesapla
        pca_temp = PCA()
        pca_temp.fit(features_scaled)
        
        # Açıklanan varyansı biriktirerek bileşen sayısını belirle
        cumulative_variance = np.cumsum(pca_temp.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= explained_variance_threshold) + 1
        n_components = min(n_components, len(feature_names))
    
    pca_model = PCA(n_components=n_components)
    pca_features = pca_model.fit_transform(features_scaled)
    
    # Sonuçları DataFrame'e çevir
    pca_columns = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(
        pca_features,
        columns=pca_columns,
        index=df.index
    )
    
    # Orijinal verilerle birleştir
    pca_df = pd.concat([
        df[['city', 'country', 'region']].reset_index(drop=True),
        pca_df.reset_index(drop=True)
    ], axis=1)
    
    return pca_df, pca_model, scaler, pca_model.explained_variance_ratio_


def apply_kmeans_clustering(df, n_clusters=None, use_pca=False, pca_components=None):
    """
    K-means clustering uygular.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Şehir verilerini içeren DataFrame
    n_clusters : int, optional
        Küme sayısı. None ise en iyi sayıyı bulmak için silhouette score kullanılır.
    use_pca : bool, default=False
        PCA kullanılıp kullanılmayacağı
    pca_components : int, optional
        PCA için bileşen sayısı (use_pca=True ise)
    
    Returns:
    --------
    clustered_df : pandas.DataFrame
        Küme etiketleri eklenmiş DataFrame
    kmeans_model : sklearn.cluster.KMeans
        Eğitilmiş K-means modeli
    scaler : sklearn.preprocessing.StandardScaler
        Kullanılan scaler
    pca_model : sklearn.decomposition.PCA or None
        Kullanılan PCA modeli (varsa)
    silhouette_avg : float
        Ortalama silhouette score
    """
    # Özellikleri hazırla
    features_df, feature_names = prepare_features_for_clustering(df)
    
    # Standardizasyon
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df)
    
    pca_model = None
    
    # İsteğe bağlı PCA
    if use_pca:
        if pca_components is None:
            pca_components = min(10, len(feature_names))
        
        pca_model = PCA(n_components=pca_components)
        features_scaled = pca_model.fit_transform(features_scaled)
        print(f"PCA ile {pca_components} bileşene indirildi.")
    
    # En iyi küme sayısını bul (n_clusters belirtilmemişse)
    if n_clusters is None:
        best_score = -1
        best_k = 2
        k_range = range(2, min(11, len(df) // 10 + 1))  # 2-10 arası veya veri boyutuna göre
        
        for k in k_range:
            kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels_temp = kmeans_temp.fit_predict(features_scaled)
            score = silhouette_score(features_scaled, labels_temp)
            
            if score > best_score:
                best_score = score
                best_k = k
        
        n_clusters = best_k
        print(f"En iyi küme sayısı: {n_clusters} (Silhouette Score: {best_score:.3f})")
    
    # K-means uygula
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans_model.fit_predict(features_scaled)
    
    # Silhouette score hesapla
    silhouette_avg = silhouette_score(features_scaled, cluster_labels)
    
    # Sonuçları DataFrame'e ekle
    clustered_df = df.copy()
    clustered_df['cluster'] = cluster_labels
    
    return clustered_df, kmeans_model, scaler, pca_model, silhouette_avg


def analyze_clusters(clustered_df):
    """
    Kümeleri analiz eder ve özet istatistikler döndürür.
    
    Parameters:
    -----------
    clustered_df : pandas.DataFrame
        Küme etiketleri eklenmiş DataFrame
    
    Returns:
    --------
    cluster_summary : pandas.DataFrame
        Her küme için özet istatistikler
    """
    if 'cluster' not in clustered_df.columns:
        raise ValueError("DataFrame'de 'cluster' sütunu bulunamadı.")
    
    # Sayısal özellikler
    numeric_cols = [
        'culture', 'adventure', 'nature', 'beaches', 'nightlife',
        'cuisine', 'wellness', 'urban', 'seclusion',
        'latitude', 'longitude', 'avg_temp_summer', 'avg_temp_winter',
        'budget_numeric', 'distance_to_airport_km'
    ]
    
    available_cols = [col for col in numeric_cols if col in clustered_df.columns]
    
    # Küme başına ortalama değerler
    cluster_summary = clustered_df.groupby('cluster')[available_cols].mean()
    cluster_summary['city_count'] = clustered_df.groupby('cluster').size()
    
    return cluster_summary


def get_cluster_characteristics(clustered_df, cluster_id):
    """
    Belirli bir kümenin özelliklerini döndürür.
    
    Parameters:
    -----------
    clustered_df : pandas.DataFrame
        Küme etiketleri eklenmiş DataFrame
    cluster_id : int
        Küme ID'si
    
    Returns:
    --------
    characteristics : dict
        Küme özellikleri
    cities : pandas.DataFrame
        Bu kümedeki şehirler
    """
    cluster_data = clustered_df[clustered_df['cluster'] == cluster_id]
    
    if len(cluster_data) == 0:
        return None, None
    
    characteristics = {
        'cluster_id': cluster_id,
        'city_count': len(cluster_data),
        'top_countries': cluster_data['country'].value_counts().head(5).to_dict(),
        'top_regions': cluster_data['region'].value_counts().head(5).to_dict(),
    }
    
    # Sayısal özelliklerin ortalamaları
    numeric_cols = [
        'culture', 'adventure', 'nature', 'beaches', 'nightlife',
        'cuisine', 'wellness', 'urban', 'seclusion',
        'avg_temp_summer', 'avg_temp_winter', 'budget_numeric'
    ]
    
    for col in numeric_cols:
        if col in cluster_data.columns:
            characteristics[f'avg_{col}'] = cluster_data[col].mean()
    
    cities = cluster_data[['city', 'country', 'region']].copy()
    
    return characteristics, cities

