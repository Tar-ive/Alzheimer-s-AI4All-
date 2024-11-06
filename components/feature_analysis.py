import streamlit as st
import pandas as pd
import plotly.express as px

def show_feature_analysis(X, y):
    st.header("Feature Analysis")
    
    # Feature statistics
    st.subheader("Feature Statistics")
    stats_df = X.describe()
    st.dataframe(stats_df)
    
    # Feature selection
    st.subheader("Feature Explorer")
    col1, col2 = st.columns(2)
    
    with col1:
        feature1 = st.selectbox(
            "Select first feature",
            options=X.columns,
            key="feature1"
        )
    
    with col2:
        feature2 = st.selectbox(
            "Select second feature",
            options=X.columns,
            key="feature2"
        )
    
    # Create scatter plot
    fig = px.scatter(
        x=X[feature1],
        y=X[feature2],
        color=y,
        title=f"{feature1} vs {feature2}",
        labels={'x': feature1, 'y': feature2, 'color': 'Diagnosis'},
        color_discrete_sequence=['#FF4B4B', '#0068C9']
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature correlation
    st.subheader("Feature Correlation")
    correlation = X[[feature1, feature2]].corr()
    st.write(f"Correlation coefficient: {correlation.iloc[0,1]:.3f}")
