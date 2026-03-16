# Real Estate Price Intelligence System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![Pandas](https://img.shields.io/badge/Data%20Analysis-Pandas-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

A Machine Learning project that predicts residential property prices using **Linear Regression** and provides insights into key factors influencing housing values. The project demonstrates a full **end‑to‑end ML pipeline** including preprocessing, training, evaluation, and deployment.

------------------------------------------------------------------------

# Project Overview

Real estate price estimation helps buyers, sellers, and investors understand market value. This system analyzes housing attributes such as living area, number of bedrooms, and construction year to estimate property price.

The project showcases:

-   Data preprocessing
-   Feature selection
-   Machine learning model training
-   Model evaluation
-   Interactive prediction dashboard

------------------------------------------------------------------------

# Features

-   House price prediction using Linear Regression
-   Handling missing values in dataset
-   Feature scaling using StandardScaler
-   Model evaluation using regression metrics
-   Interactive **Streamlit dashboard**
-   Data visualization for insights

------------------------------------------------------------------------

# Dataset

The project uses the **Ames Housing Dataset**, a well-known dataset for
real estate price prediction.

Selected model features: 

- Gr Liv Area
- Bedroom AbvGr
- Full Bath
- Garage Cars
- Overall Qual
- Year Built
- SalePrice

------------------------------------------------------------------------

# Project Structure

    real-estate-price-intelligence-system
    │
    ├── data
    │   └── housing.csv
    │
    ├── src
    │   ├── data_preprocessing.py
    │   ├── train_model.py
    │   ├── evaluate_model.py
    │   ├── feature_importance.py
    │   └── predict_price.py
    │
    ├── models
    │   ├── linear_regression_model.pkl
    │   └── scaler.pkl
    │
    ├── app
    │   └── streamlit_app.py
    │
    ├── visuals
    │   └── plots.py
    │
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

# Machine Learning Pipeline

### Data Preprocessing

-   Handle missing values using median imputation
-   Feature scaling using StandardScaler

### Model Training

-   Linear Regression model is trained to learn the relationship between
    housing attributes and sale price

### Model Evaluation

Evaluation metrics used:

-   R² Score
-   Mean Absolute Error (MAE)
-   Root Mean Squared Error (RMSE)

### Deployment

The trained model is deployed through a **Streamlit web application**
where users can input property features and get predicted prices.

------------------------------------------------------------------------

# Installation

Clone the repository

    git clone Real Estate Price Intelligence System
    cd Real Estate Price Intelligence System

Install dependencies

    pip install -r requirements.txt

------------------------------------------------------------------------

# Train the Model

    python src/train_model.py

------------------------------------------------------------------------

# Evaluate the Model

    python src/evaluate_model.py

------------------------------------------------------------------------

# Run the Application

    streamlit run app/streamlit_app.py

------------------------------------------------------------------------

# Example Prediction

Input:

  Feature           Value
  ----------------- ------------
  Living Area       2000 sq ft
  Bedrooms          3
  Bathrooms         2
  Garage Cars       2
  Overall Quality   7
  Year Built        2005

Output:

Estimated House Price: **\$285,000**

------------------------------------------------------------------------

# Technologies Used

-   Python
-   Pandas
-   NumPy
-   Scikit‑learn
-   Matplotlib
-   Seaborn
-   Streamlit

------------------------------------------------------------------------

# Future Improvements

-   Add advanced models such as Random Forest and XGBoost
-   Include location-based price prediction
-   Deploy the system on cloud platforms
-   Build an interactive real estate analytics dashboard

------------------------------------------------------------------------

# Author

**Hasini**

- Passionate about building intelligent systems and applying machine learning to real-world problems.
