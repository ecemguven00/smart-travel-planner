import streamlit as st
import pandas as pd
import altair as alt
import sys
import os
import traceback
from ui_charts import create_pca_scatter_plot, create_cluster_visualization

# --- ML MODULE IMPORT LOGIC ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from ml_nlp.feature_engineering import (
        apply_pca,
        apply_kmeans_clustering,
        analyze_clusters,
        get_cluster_characteristics
    )
    ML_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Feature Engineering modules could not be loaded. Error: {e}")
    ML_AVAILABLE = False

def show_ml_analysis_tab(filtered_df):
    """
    Displays the Machine Learning analysis tab with PCA and Clustering.
    """
    if not ML_AVAILABLE:
        st.warning("⚠️ ML Analysis module could not be loaded. Please ensure 'ml_nlp' package is installed.")
        return

    if len(filtered_df) < 3:
        st.info("ℹ️ At least 3 cities are required for ML analysis. Try loosening your filters to see more results.")
        return

    st.markdown("### 🔬 Machine Learning Analysis")
    st.markdown("Analyze your destinations using PCA (Dimensionality Reduction) and K-means Clustering.")

    # Analysis Options
    col1, col2 = st.columns(2)
    with col1:
        show_pca = st.checkbox("📉 Show PCA Analysis", value=True)
    with col2:
        show_clustering = st.checkbox("🎯 Show K-means Clustering", value=True)

    # --- PCA ANALYSIS ---
    if show_pca:
        st.markdown("---")
        st.subheader("📉 Principal Component Analysis (PCA)")
        st.markdown("PCA reduces the complexity of data to reveal underlying patterns.")

        try:
            with st.spinner("Calculating PCA..."):
                pca_df, pca_model, scaler, explained_variance = apply_pca(
                    filtered_df,
                    explained_variance_threshold=0.95
                )

            # Display PCA Stats
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Components", len(explained_variance))
            with col2:
                st.metric("Explained Variance", f"{sum(explained_variance) * 100:.2f}%")

            # Variance Table
            st.markdown("**Explained Variance Ratio by Component (First 5):**")
            variance_df = pd.DataFrame({
                'Component': [f'PC{i + 1}' for i in range(min(5, len(explained_variance)))],
                'Variance (%)': [var * 100 for var in explained_variance[:5]]
            })
            st.dataframe(variance_df, width="stretch", hide_index=True)

            # PCA Scatter Plot
            if 'PC1' in pca_df.columns and 'PC2' in pca_df.columns:
                st.markdown("**PCA Visualization (PC1 vs PC2):**")
                color_option = st.selectbox(
                    "Color By:",
                    options=['region', 'budget_level', None],
                    format_func=lambda x: {'region': 'Region', 'budget_level': 'Budget Level', None: 'None'}.get(x, 'None')
                )

                pca_scatter = create_pca_scatter_plot(pca_df, color_col=color_option)
                if pca_scatter:
                    st.altair_chart(pca_scatter, width="stretch")

                # Show raw PCA data
                if len(pca_df) > 1:
                    st.markdown("**PCA Component Values (Top 10 Cities):**")
                    display_cols = ['city', 'country', 'region'] + [f'PC{i + 1}' for i in range(min(3, len(explained_variance)))]
                    st.dataframe(pca_df[display_cols].head(10), width="stretch", hide_index=True)

        except Exception as e:
            st.error(f"Error during PCA analysis: {str(e)}")

    # --- CLUSTERING ANALYSIS ---
    if show_clustering:
        st.markdown("---")
        st.subheader("🎯 K-means Clustering")
        st.markdown("K-means groups cities with similar characteristics together.")

        col1, col2 = st.columns(2)
        with col1:
            n_clusters = st.slider("Number of Clusters", min_value=2, max_value=min(10, len(filtered_df) // 3), value=3)
        with col2:
            use_pca_for_clustering = st.checkbox("Use PCA for Clustering", value=False)

        try:
            with st.spinner("Calculating Clusters..."):
                clustered_df, kmeans_model, scaler_km, pca_model_km, silhouette_avg = apply_kmeans_clustering(
                    filtered_df,
                    n_clusters=n_clusters,
                    use_pca=use_pca_for_clustering,
                    pca_components=5 if use_pca_for_clustering else None
                )

            # Clustering Stats
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Silhouette Score", f"{silhouette_avg:.4f}",
                          help="Higher score indicates better defined clusters (Range: -1 to 1)")
            with col2:
                st.metric("Number of Clusters", n_clusters)

            # Cluster Distribution
            st.markdown("**Cluster Distribution:**")
            cluster_counts = clustered_df['cluster'].value_counts().sort_index()
            cluster_counts_df = pd.DataFrame({
                'Cluster ID': cluster_counts.index,
                'Count': cluster_counts.values
            })
            st.dataframe(cluster_counts_df, width="stretch", hide_index=True)

            # Visualization
            st.markdown("**Cluster Visualization:**")
            scatter, bar_chart = create_cluster_visualization(clustered_df)

            if scatter:
                st.altair_chart(scatter, width="stretch")

            # Cluster Characteristics
            st.markdown("**Cluster Characteristics:**")
            cluster_summary = analyze_clusters(clustered_df)

            # Filter relevant columns for summary
            important_cols = ['culture', 'adventure', 'nature', 'beaches', 'nightlife',
                              'avg_temp_summer', 'budget_numeric', 'city_count']
            available_cols = [col for col in important_cols if col in cluster_summary.columns]

            if available_cols:
                st.dataframe(cluster_summary[available_cols], width="stretch")

            # Detailed Breakdown per Cluster
            st.markdown("**Detailed Cluster Breakdown:**")
            for cluster_id in sorted(clustered_df['cluster'].unique()):
                count = len(clustered_df[clustered_df['cluster'] == cluster_id])
                with st.expander(f"🔵 Cluster {cluster_id} - {count} Cities"):
                    characteristics, cities = get_cluster_characteristics(clustered_df, cluster_id)

                    if characteristics:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Average Characteristics:**")
                            if 'avg_culture' in characteristics:
                                st.write(f"• Culture: {characteristics['avg_culture']:.1f}")
                            if 'avg_adventure' in characteristics:
                                st.write(f"• Adventure: {characteristics['avg_adventure']:.1f}")
                            if 'avg_nature' in characteristics:
                                st.write(f"• Nature: {characteristics['avg_nature']:.1f}")
                            if 'avg_beaches' in characteristics:
                                st.write(f"• Beaches: {characteristics['avg_beaches']:.1f}")

                        with col2:
                            st.write("**Most Common:**")
                            if 'top_countries' in characteristics:
                                countries = list(characteristics['top_countries'].keys())[:3]
                                st.write(f"• Countries: {', '.join(countries)}")
                            if 'top_regions' in characteristics:
                                regions = list(characteristics['top_regions'].keys())[:3]
                                st.write(f"• Regions: {', '.join(regions)}")

                        if cities is not None and len(cities) > 0:
                            st.write("**Cities in this Cluster:**")
                            st.dataframe(cities.head(10), width="stretch", hide_index=True)

        except Exception as e:
            st.error(f"Error during Clustering analysis: {str(e)}")
            st.code(traceback.format_exc())