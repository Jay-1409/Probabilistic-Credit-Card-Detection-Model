import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import numpy as np
import time
# Load the dataset
df = pd.read_csv('csvs/creditcard.csv')

# Step 1: Print the first few rows to get an idea of the data
print("Dataset preview:")
print(df.head())

# Step 2: Check the structure of the dataset
print("\nDataset structure:")
print(df.info())

# Step 3: See how many fraudulent and non-fraudulent transactions we have
print("\nClass distribution:")
print(df['Class'].value_counts())

# Step 4: Get some statistics about the data
print("\nDataset statistics:")
print(df.describe())

# Step 5: Check for missing values
print("\nMissing values in dataset:")
print(df.isnull().sum())

# Step 6: Normalize the 'Amount' column (scale it)
scaler = StandardScaler()
df['Amount_Norm'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))

# Step 7: Select more features for the HMM model (including V1 to V28, Time, and Amount_Norm)
features = ['Time', 'Amount_Norm'] + [f'V{i}' for i in range(1, 29)]

# Step 8: Prepare the feature matrix for the HMM
X = df[features].values

# Step 9: Build a Gaussian HMM model with 2 hidden states (fraud, non-fraud)
model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=100, random_state=42)

# Step 10: Fit the model to the data
model.fit(X)

# Step 11: Predict the hidden states (fraud or not fraud)
hidden_states = model.predict(X)

# Step 12: Add the hidden state predictions to the original dataframe
df['HMM_Predicted'] = hidden_states

# Step 13: Preview the result
print("\nPredicted results (first 5 rows):")
print(df[['Time', 'Amount', 'Class', 'HMM_Predicted']].head())

# Step 14: Evaluate the model performance by comparing predictions with actual fraud labels
correct_predictions = (df['Class'] == df['HMM_Predicted']).sum()
total_transactions = df.shape[0]
accuracy = correct_predictions / total_transactions * 100

print(f"\nAccuracy of HMM model: {accuracy:.2f}%")
