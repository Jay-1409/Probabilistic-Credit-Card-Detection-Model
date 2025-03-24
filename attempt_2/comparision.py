import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.stats import weibull_min, norm

# 1. Load and preprocess data
df = pd.read_csv('csvs/creditcard.csv')
scaler = StandardScaler()
df['Amount_Norm'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
features = ['Time', 'Amount_Norm'] + [f'V{i}' for i in range(1, 29)]
X = df[features].values

# 2. Split data for evaluation
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
X_train = train_df[features].values
X_test = test_df[features].values
y_test = test_df['Class']

# 3. Train the HMM model on training data (or the full dataset if unsupervised)
hmm_model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=100, random_state=42)
hmm_model.fit(X_train)
hidden_states_train = hmm_model.predict(X_train)
hidden_states_test = hmm_model.predict(X_test)

# 4. Map hidden states to labels (this step may require validation; assume state 0 is non-fraud, state 1 is fraud)
# You may compute the average 'Class' for transactions in each state on the training set and decide accordingly.
state_0 = train_df['Class'][hidden_states_train == 0].mean()
state_1 = train_df['Class'][hidden_states_train == 1].mean()
# Assuming higher mean indicates fraud
if state_0 > state_1:
    mapping = {0: 1, 1: 0}
else:
    mapping = {0: 0, 1: 1}
y_pred_hmm = [mapping[s] for s in hidden_states_test]

# 5. Evaluate HMM model
print("HMM Model Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred_hmm))
print("Precision:", precision_score(y_test, y_pred_hmm))
print("Recall:", recall_score(y_test, y_pred_hmm))
print("F1 Score:", f1_score(y_test, y_pred_hmm))

# 6. Run your probabilistic model on the test data
# Example: Define your functions for Gaussian and Weibull probability as given in your code.
def gaussian_probability(x, mu, sigma):
    return norm.pdf(x, loc=mu, scale=sigma)

def weibull_probability(x, shape, scale):
    return weibull_min.pdf(x, c=shape, scale=scale)

# For simplicity, let's assume you have pre-calculated parameters for fraud (state 2) and non-fraud (state 1)
# These would come from your calculate_variable_values() function.
params = {
    'mu_amount_s1': 50, 'sigma_amount_s1': 20, 'shape_s1': 1.5, 'scale_s1': 100,
    'mu_amount_s2': 200, 'sigma_amount_s2': 50, 'shape_s2': 1.2, 'scale_s2': 150,
    # Similarly for distance if you have such a feature, else you might ignore or set dummy values.
    'mu_distance_s1': 10, 'sigma_distance_s1': 5,
    'mu_distance_s2': 20, 'sigma_distance_s2': 10
}

# A function to classify a transaction using your probabilistic model
def classify_transaction(transaction, params, threshold=0.5):
    prob_state1 = (gaussian_probability(transaction["amount"], params['mu_amount_s1'], params['sigma_amount_s1']) *
                   weibull_probability(transaction["time"], params['shape_s1'], params['scale_s1']) *
                   gaussian_probability(transaction["distance"], params['mu_distance_s1'], params['sigma_distance_s1']))
    prob_state2 = (gaussian_probability(transaction["amount"], params['mu_amount_s2'], params['sigma_amount_s2']) *
                   weibull_probability(transaction["time"], params['shape_s2'], params['scale_s2']) *
                   gaussian_probability(transaction["distance"], params['mu_distance_s2'], params['sigma_distance_s2']))
    
    # Normalize or compare directly; here we simply choose the higher probability as the label.
    return 1 if prob_state2 > prob_state1 else 0

# For demonstration, assume your test data includes 'time', 'amount', and 'distance'
# You might need to create a consistent format for test_df. For example:
test_df['distance'] = np.random.rand(len(test_df)) * 20  # Dummy distance values
test_df['amount'] = test_df['Amount']
test_df['time'] = test_df['Time']

y_pred_prob = test_df.apply(lambda row: classify_transaction(row, params), axis=1)

# 7. Evaluate your probabilistic model
print("\nProbabilistic Model Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred_prob))
print("Precision:", precision_score(y_test, y_pred_prob))
print("Recall:", recall_score(y_test, y_pred_prob))
print("F1 Score:", f1_score(y_test, y_pred_prob))
