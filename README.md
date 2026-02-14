🛒 Store Sales Forecasting – End-to-End Time Series ML Project
## Project Summary

This project builds a complete machine learning pipeline to forecast daily retail sales across multiple stores and product categories.

The objective is to predict future sales using historical transaction data combined with external factors such as oil prices and holidays.

The project demonstrates:
Time series feature engineering
External data integration
Gradient boosting modeling
Model evaluation using RMSE
Feature importance analysis
Clean and modular ML pipeline design

## Dataset Description

Dataset Used: https://www.kaggle.com/competitions/store-sales-time-series-forecasting

The dataset contains:
Daily sales per store and product family
Promotion indicators
Store metadata (city, state, type)
Oil price data (economic indicator)
Holiday and event information
The dataset spans multiple years and includes strong seasonal patterns.

## Methodology
1️⃣ Data Loading & Optimization
Memory-safe CSV loading
Datetime conversion
Data type optimization (int16, float32, categorical encoding)
Forward-filling missing oil prices

2️⃣ Feature Engineering
The following features were created:
📅 Date Features
Year
Month
Day
Day of week

🔁 Lag Feature
lag_7 (sales from 7 days ago)
This captures weekly seasonality, which is a strong signal in retail forecasting.

🎉 Holiday Indicator
Binary flag indicating holiday presence

⛽ External Economic Feature
Daily oil price (forward-filled)

🏪 Encoded Categorical Variables
Store
Product family
City
State
Store type

🤖 Model Selection
Model used: GradientBoostingRegressor (scikit-learn)

Why Gradient Boosting?
Strong performance on structured/tabular data
Handles nonlinear relationships
Provides feature importance
Stable and interpretable

📈 Model Evaluation
Data was split using an 80/20 time-based split to preserve chronological order.
Evaluation Metric:
RMSE (Root Mean Squared Error)

Example result: Validation RMSE ≈ 450

This indicates the model successfully captures trends and weekly seasonality while handling variability in sales.

📊 Generated Outputs

After running the pipeline, the following artifacts are created:

File	                      Description
submission.csv	           Final forecast predictions
validation_plot.png	       Actual vs Predicted sales comparison
feature_importance.png	   Feature importance visualization
metrics.txt	               Validation RMSE

## Key Insights

Lag_7 is the most important feature, indicating strong weekly seasonality.
External features (holidays, oil prices) contribute moderately.
Retail sales forecasting is highly autocorrelated.
Gradient boosting effectively models nonlinear relationships.

## Project Structure
store_sales_forecasting/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── stores.csv
│   ├── oil.csv
│   └── holidays_events.csv
│
├── outputs/
│   ├── submission.csv
│   ├── validation_plot.png
│   ├── feature_importance.png
│   └── metrics.txt
│
├── main.py
└── README.md

## How to Run

Install dependencies:
pip install pandas numpy matplotlib scikit-learn

Run the pipeline:
python main.py

Outputs will be saved in the outputs/ directory.

## Skills Demonstrated
Time Series Forecasting
Feature Engineering
Data Preprocessing
Model Evaluation
Data Visualization
End-to-End ML Pipeline Development 
Clean Code Structuring