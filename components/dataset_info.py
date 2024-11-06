import streamlit as st
import pandas as pd

def show_dataset_info(metadata, variables):
    st.header("Dataset Information")
    
    # Dataset overview
    st.subheader("Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Instances", "174")
    with col2:
        st.metric("Number of Features", "451")
    with col3:
        st.metric("Missing Values", "No")
    
    # Dataset purpose
    st.subheader("Purpose")
    st.write("""
        The DARWIN dataset was created to allow researchers to improve existing machine learning 
        methodologies for the prediction of Alzheimer's disease via handwriting analysis.
    """)
    
    # Citations
    st.subheader("Citations")
    st.markdown("""
        If you use this dataset, please cite:
        
        1. N. D. Cilia, C. De Stefano, F. Fontanella, A. S. Di Freca, *An experimental protocol 
        to support cognitive impairment diagnosis by using handwriting analysis*, Procedia 
        Computer Science 141 (2018) 466–471.
        
        2. N. D. Cilia, G. De Gregorio, C. De Stefano, F. Fontanella, A. Marcelli, A. Parziale, 
        *Diagnosing Alzheimer's disease from online handwriting: A novel dataset and performance 
        benchmarking*, Engineering Applications of Artificial Intelligence, Vol. 111 (2022) 104822.
    """)
    
    # Variable information
    st.subheader("Variables Information")
    if variables is not None:
        st.dataframe(pd.DataFrame(variables))
    
    # License information
    st.subheader("License")
    st.info("""
        This dataset is licensed under a Creative Commons Attribution 4.0 International 
        (CC BY 4.0) license. This allows for the sharing and adaptation of the datasets 
        for any purpose, provided that appropriate credit is given.
    """)
