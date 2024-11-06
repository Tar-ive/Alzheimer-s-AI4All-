import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np

def show_model_prediction(X, y):
    st.header("Model Prediction")
    
    # Model information
    st.info("""
        This section demonstrates the SVM model's capability to predict Alzheimer's disease 
        based on handwriting features. The model is trained on the DARWIN dataset using 
        Support Vector Machine classification.
    """)
    
    # Train model
    @st.cache_resource
    def train_svm_model(X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = SVC(probability=True)
        model.fit(X_train_scaled, y_train)
        
        return model, scaler, X_test, y_test
    
    model, scaler, X_test, y_test = train_svm_model(X, y)
    
    # Model performance
    st.subheader("Model Performance")
    test_accuracy = model.score(scaler.transform(X_test), y_test)
    st.metric("Test Accuracy", f"{test_accuracy:.2%}")
    
    # Sample prediction
    st.subheader("Sample Prediction")
    
    # Select random sample
    if st.button("Get Random Sample"):
        sample_idx = np.random.randint(0, len(X_test))
        sample_X = X_test.iloc[sample_idx:sample_idx+1]
        sample_y = y_test.iloc[sample_idx]
        
        # Make prediction
        prediction = model.predict(scaler.transform(sample_X))
        probability = model.predict_proba(scaler.transform(sample_X))
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.write("Actual Class:", sample_y)
        with col2:
            st.write("Predicted Class:", prediction[0])
        
        st.write("Prediction Probability:", f"{max(probability[0]):.2%}")
