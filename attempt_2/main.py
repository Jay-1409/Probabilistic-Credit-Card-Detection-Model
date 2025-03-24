import numpy as np
import plotly.graph_objs as go
from scipy.stats import norm, weibull_min
from calculations import calculate_variable_values, calculate_state_1_prob, calculate_state_2_prob, gaussian_probability, weibull_probability
# Transition probabilities (A)
A = np.array([[0.7 ,  0.3  ],   # Transition from Legitimate (S1)
              [0.4 ,0.6]])  # Transition from Fraudulent (S2)

# Initial state probabilities (pi)
pi = np.array([0.9, 0.1])  # 90% chance starting in Legitimate, 10% in Fraudulent
def verdict(prev_amt, prev_time, amt, time):
    # Parameters for Gaussian distribution in each state (Legitimate and Fraudulent)
    # State 1 (Legitimate)
    variables = calculate_variable_values();
    mu_amount_s1, sigma_amount_s1 = variables['mu_amount_s1'], variables['sigma_amount_s1'];
    shape_s1, scale_s1 = variables['shape_s1'], variables['scale_s1']
    mu_distance_s1, sigma_distance_s1 = 10, 5

    # State 2 (Fraudulent)
    mu_amount_s2, sigma_amount_s2 = variables['mu_amount_s2'], variables['sigma_amount_s2']
    shape_s2, scale_s2 = variables['shape_s2'], variables['scale_s2']
    mu_distance_s2, sigma_distance_s2 = 100, 50
    
    # amount_range = np.linspace(0, 1000, 1000)
    # time_range = np.linspace(0, 100, 1000)

    # # Gaussian PDFs for Amount
    # trace_amount_s1 = go.Scatter(x=amount_range, y=norm.pdf(amount_range, mu_amount_s1, sigma_amount_s1), mode='lines', name='Legitimate - Amount')
    # trace_amount_s2 = go.Scatter(x=amount_range, y=norm.pdf(amount_range, mu_amount_s2, sigma_amount_s2), mode='lines', name='Fraudulent - Amount')

    # # Weibull PDFs for Time
    # trace_time_s1 = go.Scatter(x=time_range, y=weibull_min.pdf(time_range, shape_s1, scale=scale_s1), mode='lines', name='Legitimate - Time')
    # trace_time_s2 = go.Scatter(x=time_range, y=weibull_min.pdf(time_range, shape_s2, scale=scale_s2), mode='lines', name='Fraudulent - Time')

    # # Create subplots for both Amount and Time PDFs
    # fig = go.Figure()
    # fig.add_trace(trace_amount_s1)
    # fig.add_trace(trace_amount_s2)
    # fig.add_trace(trace_time_s1)
    # fig.add_trace(trace_time_s2)

    # fig.update_layout(
    #     title="PDFs for Amount and Time (Weibull & Gaussian)",
    #     xaxis_title="Values",
    #     yaxis_title="Probability Density",
    #     legend_title="States",
    # )

    # fig.show()
    # print(variables)
    # Example Observations
    
    transaction_1 = {"amount": prev_amt, "time": prev_time, "distance": 120}  # Observation 1
    transaction_2 = {"amount": amt, "time": time, "distance": 1}  # Observation 2



    # Compute probabilities for Transaction 1 (State 1 and State 2)
    p_s1_transaction_1 = calculate_state_1_prob(
        transaction_1,
        variables['mu_amount_s1'], variables['sigma_amount_s1'],
        variables['shape_s1'], variables['scale_s1'],
        mu_distance_s1=10, sigma_distance_s1=5  # Modify as per your data
    )

    p_s2_transaction_1 = calculate_state_2_prob(
        transaction_1,
        variables['mu_amount_s2'], variables['sigma_amount_s2'],
        variables['shape_s2'], variables['scale_s2'],
        mu_distance_s2=100, sigma_distance_s2=50  # Modify as per your data
    )

    # Compute probabilities for Transaction 2 (State 1 and State 2)
    p_s1_transaction_2 = calculate_state_1_prob(
        transaction_2,
        variables['mu_amount_s1'], variables['sigma_amount_s1'],
        variables['shape_s1'], variables['scale_s1'],
        mu_distance_s1=10, sigma_distance_s1=5  # Modify as per your data
    )

    p_s2_transaction_2 = calculate_state_2_prob(
        transaction_2,
        variables['mu_amount_s2'], variables['sigma_amount_s2'],
        variables['shape_s2'], variables['scale_s2'],
        mu_distance_s2=100, sigma_distance_s2=50  # Modify as per your data
    )
    np.set_printoptions(precision=20)
    # Display results
    print(f"Transaction 1 - Legitimate State Probability: {p_s1_transaction_1}")
    print(f"Transaction 1 - Fraudulent State Probability: {p_s2_transaction_1}")
    print(f"Transaction 2 - Legitimate State Probability: {p_s1_transaction_2}")
    print(f"Transaction 2 - Fraudulent State Probability: {p_s2_transaction_2}")
    # Initial probability for Transaction 1
    total_prob_transaction_1 = pi[0] * p_s1_transaction_1 + pi[1] * p_s2_transaction_1
    print(f"Total probability for Transaction 1: {total_prob_transaction_1}")

    # For the next transaction (Transaction 2), considering state transition from Transaction 1
    total_prob_transaction_2 = (p_s1_transaction_1 * A[0, 0] * p_s1_transaction_2) + \
                            (p_s1_transaction_1 * A[0, 1] * p_s2_transaction_2) + \
                            (p_s2_transaction_1 * A[1, 0] * p_s1_transaction_2) + \
                            (p_s2_transaction_1 * A[1, 1] * p_s2_transaction_2)
    print(f"Total probability for Transaction 2: {total_prob_transaction_2}")

    # Set the threshold for classifying a transaction as fraudulent
    fraud_threshold = 0.5  # You can adjust this threshold based on your risk tolerance
    # Function to decide if a transaction is fraudulent
    def is_fraudulent(p_s1, p_s2, threshold):
        # Normalize the probabilities
        total_prob = p_s1 + p_s2
        prob_s1 = p_s1 / total_prob
        prob_s2 = p_s2 / total_prob
        
        # Print the probabilities
        print(f"Probability of Legitimate: {prob_s1}")
        print(f"Probability of Fraudulent: {prob_s2}")
        
        # Compare the probabilities
        if prob_s2 > threshold:
            return True  # Flag as fraud
        else:
            return False  # Legitimate

    # Example decision for Transaction 1
    # fraudulent_transaction_1 = is_fraudulent(p_s1_transaction_1, p_s2_transaction_1, fraud_threshold)
    # print(f"Transaction 1 is fraudulent: {fraudulent_transaction_1}")
    # Example decision for Transaction 2
    fraudulent_transaction_2 = is_fraudulent(p_s1_transaction_2, p_s2_transaction_2, fraud_threshold)
    print(f"Transaction is fraudulent: {fraudulent_transaction_2}")
    
