# main.py
import numpy as np
import plotly.graph_objs as go
from scipy.stats import norm, weibull_min
from calculations import (
    calculate_variable_values, 
    calculate_state_1_prob, 
    calculate_state_2_prob, 
    gaussian_probability, 
    weibull_probability
)

# Original Transition probabilities (A)
ORIGINAL_A = np.array([[0.7, 0.3],   # Transition from Legitimate (S1)
                       [0.4, 0.6]])  # Transition from Fraudulent (S2)

# Initial state probabilities (pi)
pi = np.array([0.9, 0.1])  # 90% chance starting in Legitimate, 10% in Fraudulent

def get_transition_matrix(use_noisy=False, noise_level=0.05):
    """
    Returns the original or a noisy transition matrix.
    """
    A = ORIGINAL_A.copy()
    if use_noisy:
        noise = np.random.uniform(-noise_level, noise_level, size=A.shape)
        A_noisy = A + noise
        # Set any negative values to 0
        A_noisy[A_noisy < 0] = 0
        # Normalize rows so that each row sums to 1
        A_noisy = A_noisy / A_noisy.sum(axis=1, keepdims=True)
        return A_noisy
    else:
        return A

def verdict(prev_amt, prev_time, amt, cur_time, use_noisy=False):
    """
    Computes the probabilities for two transactions using either the original
    or a noisy transition matrix based on the flag use_noisy.
    """
    # Retrieve parameters from your helper function
    variables = calculate_variable_values()
    # State 1 (Legitimate) parameters
    mu_amount_s1, sigma_amount_s1 = variables['mu_amount_s1'], variables['sigma_amount_s1']
    shape_s1, scale_s1 = variables['shape_s1'], variables['scale_s1']
    mu_distance_s1, sigma_distance_s1 = 10, 5

    # State 2 (Fraudulent) parameters
    mu_amount_s2, sigma_amount_s2 = variables['mu_amount_s2'], variables['sigma_amount_s2']
    shape_s2, scale_s2 = variables['shape_s2'], variables['scale_s2']
    mu_distance_s2, sigma_distance_s2 = 100, 50
    
    # Create two sample transactions
    transaction_1 = {"amount": prev_amt, "time": prev_time, "distance": 120}
    transaction_2 = {"amount": amt, "time": cur_time, "distance": 1}

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

    # Set print options for clarity
    np.set_printoptions(precision=6)
    
    # Display state probabilities for both transactions
    print(f"Transaction 1 - Legitimate State Probability: {p_s1_transaction_1}")
    print(f"Transaction 1 - Fraudulent State Probability: {p_s2_transaction_1}")
    print(f"Transaction 2 - Legitimate State Probability: {p_s1_transaction_2}")
    print(f"Transaction 2 - Fraudulent State Probability: {p_s2_transaction_2}")
    
    # Total probability for Transaction 1 using the initial state probabilities (pi)
    total_prob_transaction_1 = pi[0] * p_s1_transaction_1 + pi[1] * p_s2_transaction_1
    print(f"Total probability for Transaction 1: {total_prob_transaction_1}")
    
    # Get the transition matrix (original or noisy based on the flag)
    A = get_transition_matrix(use_noisy)
    print("Transition Matrix used:")
    print(A)
    
    # Total probability for Transaction 2, considering state transition from Transaction 1
    total_prob_transaction_2 = (p_s1_transaction_1 * A[0, 0] * p_s1_transaction_2 +
                                p_s1_transaction_1 * A[0, 1] * p_s2_transaction_2 +
                                p_s2_transaction_1 * A[1, 0] * p_s1_transaction_2 +
                                p_s2_transaction_1 * A[1, 1] * p_s2_transaction_2)
    print(f"Total probability for Transaction 2: {total_prob_transaction_2}")
    
    # Set the threshold for flagging fraud
    fraud_threshold = 0.5

    def is_fraudulent(p_s1, p_s2, threshold):
        total = p_s1 + p_s2
        prob_legit = p_s1 / total
        prob_fraud = p_s2 / total
        print(f"Normalized Probability of Legitimate: {prob_legit}")
        print(f"Normalized Probability of Fraudulent: {prob_fraud}")
        return True if prob_fraud > threshold else False

    fraudulent = is_fraudulent(p_s1_transaction_2, p_s2_transaction_2, fraud_threshold)
    print(f"Transaction is fraudulent: {fraudulent}")

# --- Interactive Comparison Script ---

if __name__ == "__main__":
    import time as t
    testcase = int(input("ENTER THE NUMBER OF TESTCASES: "))
    while testcase > 0:
        print("\n--------------------------")
        prev_amt = float(input("ENTER THE PREVIOUS TRANSACTION AMOUNT: "))
        prev_time = float(input("ENTER THE PREVIOUS TRANSACTION TIME: "))
        amt = float(input("ENTER THE CURRENT TRANSACTION AMOUNT: "))
        cur_time = float(input("ENTER THE CURRENT TRANSACTION TIME: "))

        # Run with the original transition matrix
        print("\n--- Running with ORIGINAL Transition Matrix ---")
        start_time = t.time()
        verdict(prev_amt, prev_time, amt, cur_time, use_noisy=False)
        original_time = t.time() - start_time
        print(f"Execution time: {original_time:.4f} seconds")

        # Run with a noisy transition matrix
        print("\n--- Running with NOISY Transition Matrix ---")
        start_time = t.time()
        verdict(prev_amt, prev_time, amt, cur_time, use_noisy=True)
        noisy_time = t.time() - start_time
        print(f"Execution time: {noisy_time:.4f} seconds")

        testcase -= 1
