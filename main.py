import streamlit as st
from ucimlrepo import fetch_ucirepo
import pandas as pd
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
    darwin = fetch_ucirepo(id=732)
    return darwin.data.features, darwin.data.targets, darwin.metadata, darwin.variables

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
    st.info("Please make sure you have internet connection and required packages installed.")
