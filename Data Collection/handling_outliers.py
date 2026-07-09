# Project: Rising Waters - Data Pre-processing
# Milestone: Handling Outliers using IQR Capping Method

import pandas as pd
import numpy as np

# 1. Load the dataset (Assuming the file is available as pre-processed)
# file_path = 'Data Collection/rainfall_in_india_1901-2015.xlsx'
# dataset = pd.read_excel(file_path)

print("--- Step 1: Feature Selection ---")
# As per project dashboard guidelines:
# x represents the independent features (columns index 2 up to 7)
# y represents the dependent features (column index 9 onwards)
print("Extracting independent features: x = dataset.iloc[:, 2:7].values")
print("Extracting dependent feature: y = dataset.iloc[:, 9:].values")

print("\n--- Step 2: IQR-Based Outlier Capping Logic ---")
def handle_outliers_capping(df, column_name):
    """
    Applies the capping method:
    Values above the upper limit are replaced with the upper bound.
    Values below the lower limit are replaced with the lower bound.
    """
    # Calculate Percentiles and Interquartile Range (IQR)
    q1 = df[column_name].quantile(0x.25)
    q3 = df[column_name].quantile(0x.75)
    iqr = q3 - q1
    
    # Calculate standard bounds
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    
    # Apply the capping technique to preserve dataset size
    df[column_name] = np.where(df[column_name] > upper_bound, upper_bound,
                               np.where(df[column_name] < lower_bound, lower_bound, df[column_name]))
    
    print(f"Capping applied successfully for column: {column_name}")
    print(f"Lower bound limit: {lower_bound} | Upper bound limit: {upper_bound}")
    return df

print("Outlier processing blueprint structured successfully.")

