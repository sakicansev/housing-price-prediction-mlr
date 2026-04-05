# Housing Price Prediction — Multiple Linear Regression

## Overview
A predictive analytics study implementing Multiple Linear Regression (MLR) 
in Python to predict median housing prices using the California Housing dataset.
Built as part of the MSc Data Analytics program at the University for the Creative Arts.

**Tools:** Python, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn  
**Algorithm:** Multiple Linear Regression  
**Dataset:** California Housing Dataset (Scikit-learn built-in, 20,640 observations)  
**Paper:** See `Saki_Cansev.pdf` for the full academic write-up

## Problem Statement
Predict the median value of owner-occupied homes in California based on 8 
independent variables including median income, house age, average rooms, 
population, and geographical coordinates.

## Approach
- Data loading, cleaning, and exploratory analysis on 20,640 observations
- Visualization of feature relationships using scatter plots and pair plots
- 70/30 train-test split for model training and evaluation
- Model assessed on accuracy (RMSE, R²), scalability, and interpretability

## Key Results
- RMSE: 0.727 — average prediction error of 0.727 units on the house value scale
- R² Score: 0.597 — the model explains approximately 60% of variance in house prices
- Median income identified as the strongest predictor of house value

## Files
- `mlr_housing_analysis.py` — full Python implementation
- `Saki_Cansev_-_Q10506045_-_assignement.pdf` — academic paper with methodology and findings
