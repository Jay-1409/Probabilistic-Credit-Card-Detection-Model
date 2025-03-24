import pandas as pd
import numpy as np
from scipy.stats import weibull_min, norm
import handledata as hd
# Function to calculate the mean and standard deviation for Amount,
# and Weibull parameters (shape, scale) for Time.
def calculate_variable_values():
    # Load the CSV files
    fraud_data = pd.read_csv('attempt_2/fraudulent-data.csv')
    non_fraud_data = pd.read_csv('attempt_2/non-fraudent-data_v1.csv')
    if not isinstance(fraud_data, pd.DataFrame):
        print("fraud_data is not a DataFrame")
    if not isinstance(non_fraud_data, pd.DataFrame):
        print("non_fraud_data is not a DataFrame")

    #fraud_data = hd.scale_data(fraud_data)
    #non_fraud_data = hd.scale_data(non_fraud_data)
    # 1. Calculate mean and standard deviation for 'Amount' in fraudulent and non-fraudulent transactions
    mu_amount_s2 = fraud_data['Transaction_Amount'].mean()
    sigma_amount_s2 = fraud_data['Transaction_Amount'].std()
    mu_amount_s1 = non_fraud_data['Transaction_Amount'].mean()
    sigma_amount_s1 = non_fraud_data['Transaction_Amount'].std()

    # 2. Fit a Weibull distribution to 'Time' for both fraudulent and non-fraudulent transactions
    # Weibull parameters: shape (c), scale (loc, scale)
    
    # For fraudulent data
    shape_s2, loc_s2, scale_s2 = weibull_min.fit(fraud_data['Time_of_Transaction'], floc=0)  # floc=0 ensures scale fitting
    # For non-fraudulent data
    shape_s1, loc_s1, scale_s1 = weibull_min.fit(non_fraud_data['Time_of_Transaction'], floc=0)  # floc=0 ensures scale fitting
    print(mu_amount_s2);
    # Return the calculated values
    return {
        'mu_amount_s1': mu_amount_s1,
        'sigma_amount_s1': sigma_amount_s1,
        'shape_s1': shape_s1,
        'scale_s1': scale_s1,  # Weibull parameters for non-fraudulent time
        'mu_amount_s2': mu_amount_s2,
        'sigma_amount_s2': sigma_amount_s2,
        'shape_s2': shape_s2,
        'scale_s2': scale_s2  # Weibull parameters for fraudulent time
    }
def gaussian_probability(x, mu, sigma):
    """
    Calculate the Gaussian probability of x given mean (mu) and standard deviation (sigma).
    """
    return norm.pdf(x, loc=mu, scale=sigma)

# Weibull probability function (for Time)
def weibull_probability(x, shape, scale):
    """
    Calculate the Weibull probability of x given shape and scale parameters.
    """
    return weibull_min.pdf(x, c=shape, scale=scale)

# Function for Legitimate State (State 1)
def calculate_state_1_prob(transaction, mu_amount_s1, sigma_amount_s1, shape_s1, scale_s1, mu_distance_s1, sigma_distance_s1):
    prob_amount = gaussian_probability(transaction["amount"], mu_amount_s1, sigma_amount_s1)
    prob_time = weibull_probability(transaction["time"], shape_s1, scale_s1)
    prob_distance = gaussian_probability(transaction["distance"], mu_distance_s1, sigma_distance_s1)
    return prob_amount * prob_time * prob_distance

# Function for Fraudulent State (State 2)
def calculate_state_2_prob(transaction, mu_amount_s2, sigma_amount_s2, shape_s2, scale_s2, mu_distance_s2, sigma_distance_s2):
    prob_amount = gaussian_probability(transaction["amount"], mu_amount_s2, sigma_amount_s2)
    prob_time = weibull_probability(transaction["time"], shape_s2, scale_s2)
    prob_distance = gaussian_probability(transaction["distance"], mu_distance_s2, sigma_distance_s2)
    return prob_amount * prob_time * prob_distance
