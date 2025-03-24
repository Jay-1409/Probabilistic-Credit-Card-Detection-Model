import numpy as np
from hmmlearn import hmm
import pandas as pd
from calculations import calculate_variable_values
import timeit

# Load your data - assuming you have sequences of observations (e.g., Amount, Time)
# You need to reshape your data into sequences of observations
fraud_data = pd.read_csv('attempt_2/fraudulent-data.csv')
non_fraud_data = pd.read_csv('attempt_2/non-fraudent-data_v1.csv')

# Concatenate both datasets and extract only the relevant columns (e.g., Amount, Time)
all_data = pd.concat([fraud_data, non_fraud_data])
observations = all_data[['Transaction_Amount', 'Time_of_Transaction']].values  # Assuming Amount and Time are your observation features

# Train an HMM using hmmlearn (GaussianHMM)
# Assuming 2 states: Legitimate (S1) and Fraudulent (S2)
model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=100)
# Fit the HMM to the data
model.fit(observations)
# Extract the transition matrix (A) after training
transition_matrix = model.transmat_
innitial_transition_matrix = model.startprob_
print("Learned Transition Matrix (A):")
print(transition_matrix)
print("Learned Innitial probability Matrix (A):")
print(innitial_transition_matrix)
# hmm_time = timeit.timeit('model.fit(observations)', globals=globals(), number=1)
# print(f"HMM Training Time (High Precision): {hmm_time:.8f} seconds")

# # Measure HMM prediction time using timeit
# hmm_pred_time = timeit.timeit('model.predict(observations)', globals=globals(), number=1)
# print(f"HMM Prediction Time (High Precision): {hmm_pred_time:.8f} seconds")