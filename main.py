import streamlit as st
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from components.dataset_info import show_dataset_info
from components.feature_analysis import show_feature_analysis
from components.model_prediction import show_model_prediction
from components.visualizations import show_visualizations

# Page config
st.set_page_config(
    page_title="DARWIN Dataset Explorer",
    page_icon="🧠",
    layout="wide"
)

# Title and introduction
st.title("DARWIN Dataset Explorer")
st.markdown("""
    Explore the DARWIN dataset for Alzheimer's disease prediction through handwriting analysis.
    This dataset includes handwriting data from 174 participants for distinguishing Alzheimer's 
    disease patients from healthy individuals.
""")

# Load data
@st.cache_data
def load_data():
    try:
        darwin = fetch_ucirepo(id=732)
        X = darwin.data.features
        y = darwin.data.targets
        
        # Handle ID column
        id_column = None
        for col in X.columns:
            if 'id' in col.lower():
                id_column = col
                break
        
        if id_column:
            # Store ID column separately if needed
            ids = X[id_column].copy()
            # Drop ID column from features
            X = X.drop(columns=[id_column])
        
        # Convert all columns to numeric, replacing any errors with NaN
        X = X.apply(pd.to_numeric, errors='coerce')
        
        # Check for any remaining non-numeric columns or NaN values
        nan_columns = X.columns[X.isna().any()].tolist()
        if nan_columns:
            raise ValueError(f"The following columns contain non-numeric values: {nan_columns}")
        
        return X, y, darwin.metadata, darwin.variables
    except Exception as e:
        raise Exception(f"Error processing data: {str(e)}")

try:
    X, y, metadata, variables = load_data()
    
    # Main navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["Dataset Information", "Feature Analysis", "Visualizations", "Model Prediction"]
    )
    
    # Page routing
    if page == "Dataset Information":
        show_dataset_info(metadata, variables)
    elif page == "Feature Analysis":
        show_feature_analysis(X, y)
    elif page == "Visualizations":
        show_visualizations(X, y)
    elif page == "Model Prediction":
        show_model_prediction(X, y)
        
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.info("Please ensure all features are numeric and properly formatted. Check the error message for details about problematic columns.")
