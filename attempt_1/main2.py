import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('creditcard.csv')

# Print the first few rows to get an idea of the data
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Normalize the 'Amount' column (scale it)
scaler = StandardScaler()
df['Amount_Norm'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))

# Select only the 'Time', 'Amount_Norm', and 'Class' columns for simplicity
data = df[['Time', 'Amount_Norm', 'Class']]

# Prepare the features 'Time' and 'Amount_Norm' for the HMM
X = df[['Time', 'Amount_Norm']].values

# Custom HMM implementation parameters
n_states = 2  # Fraud and non-fraud states
n_iterations = 100  # Baum-Welch iterations

# Step 1: Randomly initialize transition matrix (A) and emission probabilities (means, variances)
np.random.seed(42)
A = np.random.rand(n_states, n_states)
A = A / A.sum(axis=1, keepdims=True)  # Normalize to ensure rows sum to 1

# Random initialization for emission probabilities (Gaussian means and variances)
means = np.random.rand(n_states, X.shape[1])
variances = np.random.rand(n_states, X.shape[1]) + 1e-5  # Avoid zero variance

# Step 2: Implement Baum-Welch algorithm

# E-step: Calculate forward probabilities
def forward(X, A, means, variances):
    N, _ = X.shape
    alpha = np.zeros((N, n_states))
    
    for i in range(n_states):
        alpha[0, i] = gaussian_prob(X[0], means[i], variances[i])
    
    for t in range(1, N):
        for j in range(n_states):
            alpha[t, j] = np.sum(alpha[t-1] * A[:, j]) * gaussian_prob(X[t], means[j], variances[j])
    
    return alpha

# M-step: Update parameters using the forward-backward probabilities
def m_step(X, alpha, means, variances):
    N, D = X.shape
    gamma = alpha / alpha.sum(axis=1, keepdims=True)
    
    for i in range(n_states):
        means[i] = np.sum(gamma[:, i].reshape(-1, 1) * X, axis=0) / np.sum(gamma[:, i])
        variances[i] = np.sum(gamma[:, i].reshape(-1, 1) * (X - means[i]) ** 2, axis=0) / np.sum(gamma[:, i])
    
    # Add random perturbation to A
    A += np.random.rand(n_states, n_states) * 0.01
    A = A / A.sum(axis=1, keepdims=True)
    
    return A, means, variances

# Gaussian probability density function
import numpy as np

# Multivariate Gaussian probability density function
def gaussian_prob(x, mean, var):
    # Ensure variance is not zero
    var = np.maximum(var, 1e-6)
    
    # Calculate the exponent
    exponent = -0.5 * np.sum(((x - mean) ** 2) / var)
    
    # Multivariate Gaussian: constant factor
    const = np.prod(1.0 / np.sqrt(2 * np.pi * var))
    
    return const * np.exp(exponent)

# Now the forward function will work with multivariate Gaussian


# Train the HMM using Baum-Welch
for iteration in range(n_iterations):
    alpha = forward(X, A, means, variances)
    A, means, variances = m_step(X, alpha, means, variances)

# Step 3: Predict hidden states using the trained model
def predict(X, A, means, variances):
    alpha = forward(X, A, means, variances)
    states = np.argmax(alpha, axis=1)
    return states

# Predict hidden states for fraud detection
hidden_states = predict(X, A, means, variances)

# Add the custom predicted states to the dataframe
df['HMM_Predicted_Custom'] = hidden_states

# Preview the results
print(df[['Time', 'Amount', 'Class', 'HMM_Predicted_Custom']].head())

# Evaluate the accuracy
accuracy = np.mean(df['Class'] == df['HMM_Predicted_Custom'])
print(f'Custom HMM Accuracy: {accuracy * 100:.2f}%')
