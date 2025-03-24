import numpy as np
import plotly.graph_objs as go
from scipy.stats import norm, weibull_min
from calculations import calculate_variable_values, calculate_state_1_prob, calculate_state_2_prob, gaussian_probability, weibull_probability

# Original Transition probabilities (A)
a = np.random.rand()
b = np.random.rand()
c = np.random.rand()
d = np.random.rand()
e = np.random.rand()
f = np.random.rand()
g = np.random.rand()
A = np.array([[a, b],   # Transition from Legitimate (S1)
              [c, d]])  # Transition from Fraudulent (S2)

# Initial state probabilities (pi)
pi = np.array([1, 0])  # 90% chance starting in Legitimate, 10% in Fraudulent

def verdict(prev_amt, prev_time, amt, time):
    # Calculate parameters for the probability functions
    variables = calculate_variable_values()
    # Parameters for State 1 (Legitimate)
    mu_amount_s1, sigma_amount_s1 = variables['mu_amount_s1'], variables['sigma_amount_s1']
    shape_s1, scale_s1 = variables['shape_s1'], variables['scale_s1']
    mu_distance_s1, sigma_distance_s1 = 10, 5

    # Parameters for State 2 (Fraudulent)
    mu_amount_s2, sigma_amount_s2 = variables['mu_amount_s2'], variables['sigma_amount_s2']
    shape_s2, scale_s2 = variables['shape_s2'], variables['scale_s2']
    mu_distance_s2, sigma_distance_s2 = 100, 50
    
    # Define two example transactions
    transaction_1 = {"amount": prev_amt, "time": prev_time, "distance": 120}  # Observation 1
    transaction_2 = {"amount": amt, "time": time, "distance": 1}              # Observation 2

    # Compute probabilities for Transaction 1 (for both states)
    p_s1_transaction_1 = calculate_state_1_prob(
        transaction_1,
        mu_amount_s1, sigma_amount_s1,
        shape_s1, scale_s1,
        mu_distance_s1, sigma_distance_s1
    )

    p_s2_transaction_1 = calculate_state_2_prob(
        transaction_1,
        mu_amount_s2, sigma_amount_s2,
        shape_s2, scale_s2,
        mu_distance_s2, sigma_distance_s2
    )

    # Compute probabilities for Transaction 2 (for both states)
    p_s1_transaction_2 = calculate_state_1_prob(
        transaction_2,
        mu_amount_s1, sigma_amount_s1,
        shape_s1, scale_s1,
        mu_distance_s1, sigma_distance_s1
    )

    p_s2_transaction_2 = calculate_state_2_prob(
        transaction_2,
        mu_amount_s2, sigma_amount_s2,
        shape_s2, scale_s2,
        mu_distance_s2, sigma_distance_s2
    )

    np.set_printoptions(precision=20)
    # Display individual state probabilities for both transactions
    # print(f"Transaction 1 - Legitimate State Probability: {p_s1_transaction_1}")
    # print(f"Transaction 1 - Fraudulent State Probability: {p_s2_transaction_1}")
    # print(f"Transaction 2 - Legitimate State Probability: {p_s1_transaction_2}")
    # print(f"Transaction 2 - Fraudulent State Probability: {p_s2_transaction_2}")
    
    # Total probability for Transaction 1 using the initial state probabilities (pi)
    total_prob_transaction_1 = pi[0] * p_s1_transaction_1 + pi[1] * p_s2_transaction_1
    # print(f"Total probability for Transaction 1: {total_prob_transaction_1}")

    total_prob_transaction_2 = (
        p_s1_transaction_1 * A[0, 0] * p_s1_transaction_2 +
        p_s1_transaction_1 * A[0, 1] * p_s2_transaction_2 +
        p_s2_transaction_1 * A[1, 0] * p_s1_transaction_2 +
        p_s2_transaction_1 * A[1, 1] * p_s2_transaction_2
    )
    # print(f"Total probability for Transaction 2: {total_prob_transaction_2}")

    # Set the threshold for classifying a transaction as fraudulent
    fraud_threshold = 0.5  # Adjust this threshold based on your risk tolerance

    def is_fraudulent(p_s1, p_s2, threshold):
        # Normalize the probabilities
        total_prob = p_s1 + p_s2
        prob_s1 = p_s1 / total_prob
        prob_s2 = p_s2 / total_prob
        # print(f"Probability of Legitimate: {prob_s1}")
        # print(f"Probability of Fraudulent: {prob_s2}")
        return True if prob_s2 > threshold else False
    fraudulent_transaction_2 = is_fraudulent(p_s1_transaction_2, p_s2_transaction_2, fraud_threshold)
    return fraudulent_transaction_2
    # Decision for Transaction 2 (for example)
    # print(f"Transaction is fraudulent: {fraudulent_transaction_2}")