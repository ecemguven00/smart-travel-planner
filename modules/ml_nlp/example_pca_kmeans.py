"""
PCA ve K-means Clustering Örnek Kullanımı

Bu script, şehir verileri üzerinde PCA ve K-means clustering'in nasıl kullanılacağını gösterir.
"""

import sys
import os

# Path'leri ayarla
current_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.join(current_dir, '../frontend')
sys.path.insert(0, frontend_dir)
sys.path.insert(0, current_dir)

from feature_engineering import (
    apply_pca,
    apply_kmeans_clustering,
    analyze_clusters,
    get_cluster_characteristics
)
from data_manager import load_data
import pandas as pd


def main():
    print("=" * 60)
    print("PCA ve K-means Clustering Örneği")
    print("=" * 60)
    
    # Veriyi yükle
    print("\n1. Veri yükleniyor...")
    data_file = os.path.join(current_dir, '../frontend/Worldwide_Travel_Cities_WithAirport_Precipitation.csv')
    
    # Dosya yolunu mutlak yola çevir
    data_file = os.path.abspath(data_file)
    
    if not os.path.exists(data_file):
        print(f"Hata: Veri dosyası bulunamadı: {data_file}")
        return
    
    # load_data fonksiyonunu kullan (artık Streamlit olmadan da çalışıyor)
    df = load_data(data_file)
    
    if df.empty:
        print("Hata: Veri yüklenemedi!")
        return
    
    print(f"Toplam {len(df)} şehir verisi yüklendi.")
    
    # PCA Uygulaması
    print("\n" + "=" * 60)
    print("2. PCA (Principal Component Analysis) Uygulanıyor...")
    print("=" * 60)
    
    pca_df, pca_model, scaler, explained_variance = apply_pca(
        df,
        explained_variance_threshold=0.95
    )
    
    print(f"\nToplam {len(explained_variance)} bileşen oluşturuldu.")
    print(f"İlk 5 bileşenin açıklanan varyans oranları:")
    for i, var in enumerate(explained_variance[:5]):
        print(f"  PC{i+1}: {var:.4f} ({var*100:.2f}%)")
    
    print(f"\nToplam açıklanan varyans: {sum(explained_variance):.4f} ({sum(explained_variance)*100:.2f}%)")
    print(f"\nPCA sonuçlarının ilk 5 satırı:")
    print(pca_df[['city', 'country'] + [f'PC{i+1}' for i in range(min(3, len(explained_variance)))]].head())
    
    # K-means Clustering (PCA olmadan)
    print("\n" + "=" * 60)
    print("3. K-means Clustering Uygulanıyor (PCA olmadan)...")
    print("=" * 60)
    
    clustered_df, kmeans_model, scaler_km, pca_model_km, silhouette_avg = apply_kmeans_clustering(
        df,
        n_clusters=None,  # Otomatik en iyi sayıyı bul
        use_pca=False
    )
    
    print(f"\nSilhouette Score: {silhouette_avg:.4f}")
    print(f"\nKüme dağılımı:")
    print(clustered_df['cluster'].value_counts().sort_index())
    
    # Küme analizi
    print("\n" + "=" * 60)
    print("4. Küme Analizi")
    print("=" * 60)
    
    cluster_summary = analyze_clusters(clustered_df)
    print("\nKüme özet istatistikleri:")
    print(cluster_summary)
    
    # Her küme için detaylı analiz
    print("\n" + "=" * 60)
    print("5. Küme Özellikleri")
    print("=" * 60)
    
    for cluster_id in sorted(clustered_df['cluster'].unique()):
        print(f"\n--- Küme {cluster_id} ---")
        characteristics, cities = get_cluster_characteristics(clustered_df, cluster_id)
        
        if characteristics:
            print(f"Şehir sayısı: {characteristics['city_count']}")
            print(f"En çok görülen ülkeler: {list(characteristics['top_countries'].keys())[:3]}")
            print(f"En çok görülen bölgeler: {list(characteristics['top_regions'].keys())[:3]}")
            
            if 'avg_culture' in characteristics:
                print(f"Ortalama kültür skoru: {characteristics['avg_culture']:.2f}")
            if 'avg_adventure' in characteristics:
                print(f"Ortalama macera skoru: {characteristics['avg_adventure']:.2f}")
            if 'avg_nature' in characteristics:
                print(f"Ortalama doğa skoru: {characteristics['avg_nature']:.2f}")
            if 'avg_beaches' in characteristics:
                print(f"Ortalama plaj skoru: {characteristics['avg_beaches']:.2f}")
            
            print(f"\nÖrnek şehirler:")
            print(cities.head(5).to_string(index=False))
    
    # K-means Clustering (PCA ile)
    print("\n" + "=" * 60)
    print("6. K-means Clustering Uygulanıyor (PCA ile)...")
    print("=" * 60)
    
    clustered_df_pca, kmeans_model_pca, scaler_pca, pca_model_pca, silhouette_avg_pca = apply_kmeans_clustering(
        df,
        n_clusters=5,  # Sabit küme sayısı
        use_pca=True,
        pca_components=5
    )
    
    print(f"\nSilhouette Score (PCA ile): {silhouette_avg_pca:.4f}")
    print(f"\nKüme dağılımı (PCA ile):")
    print(clustered_df_pca['cluster'].value_counts().sort_index())
    
    # Sonuçları kaydet
    print("\n" + "=" * 60)
    print("7. Sonuçlar kaydediliyor...")
    print("=" * 60)
    
    output_dir = os.path.join(os.path.dirname(__file__), '../../output')
    os.makedirs(output_dir, exist_ok=True)
    
    pca_df.to_csv(os.path.join(output_dir, 'pca_results.csv'), index=False)
    clustered_df.to_csv(os.path.join(output_dir, 'kmeans_clusters.csv'), index=False)
    clustered_df_pca.to_csv(os.path.join(output_dir, 'kmeans_clusters_pca.csv'), index=False)
    
    print(f"Sonuçlar '{output_dir}' klasörüne kaydedildi.")
    print("\n" + "=" * 60)
    print("Tamamlandı!")
    print("=" * 60)


if __name__ == "__main__":
    main()

