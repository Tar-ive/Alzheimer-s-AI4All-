# DARWIN Dataset Explorer 🧠

A Streamlit-based web application for exploring and analyzing the DARWIN dataset, which contains handwriting data used for Alzheimer's disease prediction.

## Overview

This application provides an interactive interface to explore handwriting data from 174 participants, used to distinguish Alzheimer's disease patients from healthy individuals. The project utilizes machine learning techniques and provides various visualization and analysis tools.

## Dataset Information

The DARWIN dataset includes:
- 174 participants
- 451 features
- Binary classification task (Healthy vs. Alzheimer's)
- No missing values
- Created for Alzheimer's disease prediction via handwriting analysis
- Published on 5/11/2022
- Licensed under Creative Commons Attribution 4.0 International (CC BY 4.0)

### Key Variables Include:
- Air time measurements
- Display indices
- GMRT (in air and on paper)
- Maximum x/y extensions
- Mean acceleration (in air and on paper)
- Mean GMRT

## Technical Implementation

The project implements several machine learning approaches:
1. **Data Preprocessing**
   - Feature standardization using StandardScaler
   - Dimensionality reduction using PCA
   - Train-test splitting (80-20)

2. **Models Implemented**
   - Random Forest Classifier
   - Support Vector Machines (SVM)
     - RBF Kernel
     - Linear Kernel
   - Logistic Regression

3. **Visualization**
   - Pre-PCA data visualization
   - Post-PCA scatter plots
   - Decision boundary plots for each model
   - Feature importance analysis

4. **Model Persistence**
   - Trained models saved using pickle
   - SVM model selected as final implementation

## Features

- **Dataset Information**
  - Complete overview of the DARWIN dataset
  - Dataset statistics and metadata
  - Licensing and citation information

- **Feature Analysis**
  - Interactive feature exploration
  - Statistical analysis of individual features
  - Feature importance visualization

- **Advanced Analysis**
  - Group-wise statistical comparisons
  - T-test analysis for features
  - Descriptive statistics by diagnostic groups

- **Visualizations**
  - Feature distribution plots
  - PCA visualization
  - Interactive plots using Plotly

- **Model Prediction**
  - Real-time predictions using top features
  - Interactive input for feature values
  - Model performance metrics

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

Required dependencies (as specified in pyproject.toml):
* plotly >= 5.24.1
* scikit-learn >= 1.5.2
* scipy >= 1.14.1
* seaborn >= 0.13.2
* streamlit >= 1.39.0
* ucimlrepo >= 0.0.7
* sklearn-lvq
* reportlab

## Usage

Run the application using:
```bash
streamlit run main.py --server.port 5000
```

The application will be available at `http://localhost:5000`

### Using the Dataset in Python
```python
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
darwin = fetch_ucirepo(id=732) 
  
# data (as pandas dataframes) 
X = darwin.data.features 
y = darwin.data.targets 
```

## Project Structure

* main.py: Main application entry point
* darwin.py: Core implementation of machine learning models
* `components/`
  * dataset_info.py: Dataset information display
  * model_prediction.py: Real-time prediction interface
  * visualizations.py: Data visualization components
  * advanced_analysis.py: Statistical analysis tools

## Development

Developed for AI4All Ignite.

Frontend Streamlit Application developed by Saksham Adhikari
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Saksham_Adhikari-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/saksham-adhikari-4727571b5/)

## License

This project is open-source. The DARWIN dataset is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).

## Citations

When using this dataset, please cite:

1. N. D. Cilia, C. De Stefano, F. Fontanella, A. S. Di Freca, "An experimental protocol to support cognitive impairment diagnosis by using handwriting analysis"

2. N. D. Cilia, G. De Gregorio, C. De Stefano, F. Fontanella, A. Marcelli, A. Parziale, "Diagnosing Alzheimer's disease from online handwriting: A novel dataset and performance benchmarking"

3. Fontanella, F. (2022). DARWIN [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55D0K.
