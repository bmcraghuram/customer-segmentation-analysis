"""
Step 1: Data cleaning and feature engineering
Mirrors the original capstone pipeline: clean -> encode -> prep for correlation/PCA
"""
import pandas as pd
import numpy as np
import datetime as dt

df = pd.read_csv('marketing_campaign.csv', sep='\t')
print(f"Raw shape: {df.shape}")

# Drop rows with missing Income (mirrors original's null-handling on usersid)
df = df.dropna(subset=['Income'])
print(f"After dropping missing Income: {df.shape}")

# Remove obvious outliers (data-entry errors), same spirit as the original's outlier note
df = df[df['Income'] < 200000]
df = df[df['Year_Birth'] > 1930]

# Feature engineering
df['Age'] = 2015 - df['Year_Birth']  # dataset collected ~2014-2015 per enrollment dates
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')
df['Tenure_Days'] = (df['Dt_Customer'].max() - df['Dt_Customer']).dt.days

# Total spend and total purchases (aggregate behavioral features, analogous to SalesBand/OrdersBand)
spend_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
purchase_cols = ['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
df['TotalSpend'] = df[spend_cols].sum(axis=1)
df['TotalPurchases'] = df[purchase_cols].sum(axis=1)

# Simplify Marital_Status categories (raw data has noisy labels like 'Alone','Absurd','YOLO')
df['Marital_Status'] = df['Marital_Status'].replace(
    {'Alone': 'Single', 'Absurd': 'Single', 'YOLO': 'Single'}
)

# Drop redundant/id columns before correlation & modeling
drop_cols = ['ID', 'Year_Birth', 'Dt_Customer', 'Z_CostContact', 'Z_Revenue']
df_model = df.drop(columns=drop_cols)

df_model.to_csv('df_model.csv', index=False)
print(f"Final modeling shape: {df_model.shape}")
print(df_model.dtypes)
