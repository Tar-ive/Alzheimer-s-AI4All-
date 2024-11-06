import streamlit as st
import pandas as pd
import numpy as np
from components.dataset_info import show_dataset_info
from components.feature_analysis import show_feature_analysis
from components.model_prediction import show_model_prediction
from components.visualizations import show_visualizations
from components.advanced_analysis import show_advanced_analysis
from components.feature_analysis import show_feature_analysis


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
        # Read the CSV file directly
        data = pd.read_csv('data.csv')
        
        # Separate features and target
        if 'target' in data.columns:
            y = data['target']
            X = data.drop('target', axis=1)
        else:
            # If target is the last column
            y = data.iloc[:, -1]
            X = data.iloc[:, :-1]
        
        # Convert y to 1D array
        y = y.values.ravel()
        
        # Remove any ID columns (usually first column)
        if 'id' in X.columns.str.lower():
            X = X.drop(columns=[col for col in X.columns if 'id' in col.lower()])
        
        # Convert all remaining columns to float
        X = X.astype(float)
        
        return X, y, None, None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None

# Load and process data
X, y, metadata, variables = load_data()

if X is not None and y is not None:
    # Main navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["Dataset Information", "Feature Analysis", "Advanced Analysis", "Visualizations", "Model Prediction"]
    )
    
    # Page routing
    if page == "Dataset Information":
        show_dataset_info(metadata, variables)
    elif page == "Feature Analysis":
        show_feature_analysis(X, y)
    elif page == "Advanced Analysis":
        show_advanced_analysis(X, y)
    elif page == "Visualizations":
        show_visualizations(X, y)
    elif page == "Model Prediction":
        show_model_prediction(X, y)
else:
    st.error("Failed to load the dataset. Please check if the data.csv file exists and is properly formatted.")
    st.info("The data.csv file should contain features and a target column, with all feature columns being numeric.")
