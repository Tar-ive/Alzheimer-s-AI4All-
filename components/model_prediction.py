import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import numpy as np
import plotly.graph_objects as go

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
        
        return model, scaler, X_test, y_test, X_train_scaled, y_train
    
    model, scaler, X_test, y_test, X_train_scaled, y_train = train_svm_model(X, y)
    
    # Model performance
    st.subheader("Model Performance")
    test_accuracy = model.score(scaler.transform(X_test), y_test)
    st.metric("Test Accuracy", f"{test_accuracy:.2%}")
    
    # Feature Importance
    st.subheader("Feature Importance Analysis")
    
    @st.cache_resource
    def calculate_feature_importance(model, X_train_scaled, y_train, X_columns):
        # Calculate permutation importance
        r = permutation_importance(model, X_train_scaled, y_train,
                                 n_repeats=10, random_state=42)
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': X_columns,
            'Importance': r.importances_mean
        })
        importance_df = importance_df.sort_values('Importance', ascending=True)
        
        return importance_df
    
    importance_df = calculate_feature_importance(model, X_train_scaled, y_train, X.columns)
    
    # Display top 15 most important features
    top_n = 15
    top_features = importance_df.tail(top_n)
    
    fig = go.Figure(go.Bar(
        x=top_features['Importance'],
        y=top_features['Feature'],
        orientation='h',
        marker=dict(color='#FF4B4B')
    ))
    
    fig.update_layout(
        title=f'Top {top_n} Most Important Features',
        xaxis_title='Permutation Importance',
        yaxis_title='Feature',
        height=600,
        yaxis={'categoryorder':'total ascending'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Additional information about feature importance
    st.info("""
        Feature importance is calculated using permutation importance, which measures how much 
        the model performance decreases when a feature is randomly shuffled. Higher values 
        indicate more important features for the model's predictions.
    """)
    
    # Sample prediction
    st.subheader("Sample Prediction")
    
    # Select random sample
    if st.button("Get Random Sample"):
        sample_idx = np.random.randint(0, len(X_test))
        sample_X = X_test[sample_idx:sample_idx+1]
        sample_y = y_test[sample_idx]
        
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
