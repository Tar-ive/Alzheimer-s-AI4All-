import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import numpy as np

def show_visualizations(X, y):
    st.header("Data Visualizations")
    
    # Feature distribution
    st.subheader("Feature Distribution")
    feature = st.selectbox(
        "Select feature to visualize",
        options=X.columns
    )
    
    # Create distribution plot
    fig = px.histogram(
        X,
        x=feature,
        color=y,
        marginal="box",
        title=f"Distribution of {feature}",
        labels={'color': 'Diagnosis'},
        color_discrete_sequence=['#FF4B4B', '#0068C9']
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # PCA visualization
    st.subheader("PCA Visualization")
    
    @st.cache_data
    def perform_pca(X):
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        return X_pca, pca.explained_variance_ratio_
    
    X_pca, exp_var = perform_pca(X)
    
    fig_pca = px.scatter(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        color=y,
        title="PCA Visualization (First 2 Components)",
        labels={
            'x': f'PC1 ({exp_var[0]:.2%} explained variance)',
            'y': f'PC2 ({exp_var[1]:.2%} explained variance)',
            'color': 'Diagnosis'
        },
        color_discrete_sequence=['#FF4B4B', '#0068C9']
    )
    st.plotly_chart(fig_pca, use_container_width=True)
