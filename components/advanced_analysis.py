import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope

def show_advanced_analysis(X, y):
    st.header("Advanced Statistical Analysis")
    
    # Convert y to pandas series if it's numpy array
    y = pd.Series(y, name='diagnosis')
    
    # Create a DataFrame with both features and target
    df = pd.concat([X, y], axis=1)
    
    # Descriptive Statistics by Group
    st.subheader("Descriptive Statistics by Group")
    
    # Select feature for analysis
    feature = st.selectbox(
        "Select feature for detailed analysis",
        options=X.columns
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Group statistics
        group_stats = df.groupby('diagnosis')[feature].describe()
        st.write("Group Statistics:")
        st.dataframe(group_stats)
        
        # Statistical Test
        healthy = df[df['diagnosis'] == 0][feature]
        alzheimer = df[df['diagnosis'] == 1][feature]
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(healthy, alzheimer)
        
        st.write("Independent T-Test Results:")
        st.write(f"t-statistic: {t_stat:.4f}")
        st.write(f"p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            st.success("Statistically significant difference between groups (p < 0.05)")
        else:
            st.warning("No statistically significant difference between groups (p ≥ 0.05)")
    
    with col2:
        # Create box plot
        fig = px.box(df, x='diagnosis', y=feature, 
                    color='diagnosis',
                    title=f'Distribution of {feature} by Group',
                    labels={'diagnosis': 'Diagnosis', 'y': feature},
                    color_discrete_sequence=['#0068C9', '#FF4B4B'])
        st.plotly_chart(fig)
    
    # Outlier Analysis
    st.subheader("Outlier Analysis")
    
    # Compute outliers using Elliptic Envelope
    outlier_detector = EllipticEnvelope(contamination=0.1, random_state=42)
    outliers = outlier_detector.fit_predict(X[[feature]])
    
    # Plot outliers
    fig_outliers = px.scatter(
        x=range(len(X)),
        y=X[feature],
        color=outliers == -1,
        title=f'Outlier Detection for {feature}',
        labels={'x': 'Sample Index', 'y': feature, 'color': 'Is Outlier'},
        color_discrete_sequence=['#0068C9', '#FF4B4B']
    )
    st.plotly_chart(fig_outliers)
    
    # Correlation Analysis
    st.subheader("Feature Correlation Analysis")
    
    # Select number of features for correlation
    n_features = st.slider("Select number of features for correlation analysis", 
                          min_value=5, max_value=20, value=10)
    
    # Get top n_features by variance
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    top_features = X_scaled.var().nlargest(n_features).index
    
    # Calculate correlation matrix
    corr_matrix = X[top_features].corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        title=f'Correlation Heatmap (Top {n_features} Features by Variance)',
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig)
    
    # Distribution Analysis
    st.subheader("Distribution Analysis")
    
    selected_features = st.multiselect(
        "Select features for distribution analysis",
        options=X.columns,
        default=list(X.columns[:3])
    )
    
    if selected_features:
        # Create distplot for selected features
        for feat in selected_features:
            fig = px.histogram(
                X, x=feat,
                marginal='box',
                title=f'Distribution of {feat}',
                color=y,
                labels={'color': 'Diagnosis'},
                color_discrete_sequence=['#0068C9', '#FF4B4B']
            )
            st.plotly_chart(fig)
            
            # Add skewness and kurtosis
            skew = stats.skew(X[feat])
            kurt = stats.kurtosis(X[feat])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Skewness", f"{skew:.3f}")
            with col2:
                st.metric("Kurtosis", f"{kurt:.3f}")
            
            st.markdown("---")
