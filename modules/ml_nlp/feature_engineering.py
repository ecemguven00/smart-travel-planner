import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os
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

def plot_correlation_matrix(df, save_path=None):
    numeric_df = df.select_dtypes(include=['number'])

    corr = numeric_df.corr()

    plt.figure(figsize=(14, 10))
    sns.heatmap(
        corr,
        cmap='coolwarm',
        center=0,
        linewidths=0.5
    )
    plt.title("Feature Correlation Matrix", fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    plt.close()
    
def plot_cumulative_variance(explained_variance, save_path=None):
    cumulative_variance = np.cumsum(explained_variance)

    plt.figure(figsize=(8, 6))
    plt.plot(
        range(1, len(cumulative_variance) + 1),
        cumulative_variance,
        marker='o'
    )
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA – Cumulative Explained Variance")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    plt.close()
    
def plot_elbow_method(df, use_pca=True, pca_components=11, max_k=15, save_path=None):
    """
    df : pandas.DataFrame
        Veri seti
    use_pca : bool
        PCA kullanılsın mı? (Genelde True olmalı)
    pca_components : int
        PCA bileşen sayısı
    max_k : int
        Denenecek maksimum küme sayısı (Genelde 10-15 yeterli)
    save_path : str
        Grafiğin kaydedileceği yol
    """
    
    
    print(f"Elbow analizi yapılıyor (1'den {max_k}'e kadar deneniyor)...")
    
    # 1. Veriyi ve Ayarları Hazırla
    features_df, _ = prepare_features_for_clustering(df)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df)
    
    if use_pca:
        pca = PCA(n_components=pca_components)
        features_scaled = pca.fit_transform(features_scaled)
    
    # 2. Döngüye Gir: Her K değerini dene 1, 2, 3... 15
    inertias = []
    k_range = range(1, max_k + 1)
    
    for k in k_range:
        # random_state=42 sayesinde sonuçlar hep aynı çıkar
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features_scaled)
        inertias.append(kmeans.inertia_) # Hata değerini kaydet
        
    # 3. Grafiği Çiz
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bo-', markersize=8)
    plt.xlabel('Küme Sayısı (k)')
    plt.ylabel('Hata Değeri (Inertia)')
    plt.title(f'Elbow Method Analizi (PCA Bileşen: {pca_components})')
    plt.grid(True)
    plt.xticks(k_range)
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Grafik kaydedildi: {save_path}")
    else:
        plt.show()
    
    plt.close()