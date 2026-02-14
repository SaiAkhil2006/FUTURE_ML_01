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

## Business Application & Planning Impact
This forecasting model helps businesses in the following ways:

1️⃣ Inventory Planning
Avoid overstocking slow-moving products
Prevent stockouts during high-demand periods
Optimize warehouse space
Example:
If the model predicts higher beverage sales next weekend, the store can increase inventory beforehand.

2️⃣ Workforce Planning
Schedule more staff on high-demand days
Reduce labor costs during low-demand periods
Example:
If weekend sales are forecasted to increase, managers can adjust staffing schedules accordingly.

3️⃣ Revenue & Budget Forecasting
Estimate upcoming revenue
Plan financial targets
Manage cash flow
The forecast gives an early view of expected sales performance.

4️⃣ Promotion Strategy
Identify which products respond to promotions
Plan discounts during slow sales periods
Forecast sales uplift during holidays

5️⃣ Supply Chain Optimization
Inform suppliers about expected demand
Reduce emergency restocking
Improve delivery scheduling

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