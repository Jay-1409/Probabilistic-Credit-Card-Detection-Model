import pandas as pd
import numpy as np
from scipy.stats import weibull_min, norm
import plotly.graph_objs as go

# -----------------------------------------
# Existing Functions for Calculations
# -----------------------------------------
def calculate_variable_values():
    # Load the CSV files
    fraud_data = pd.read_csv('csvs/fraudulent-data.csv')
    non_fraud_data = pd.read_csv('csvs/non-fraudent-data_v1.csv')
    if not isinstance(fraud_data, pd.DataFrame):
        print("fraud_data is not a DataFrame")
    if not isinstance(non_fraud_data, pd.DataFrame):
        print("non_fraud_data is not a DataFrame")

    # Calculate mean and standard deviation for 'Transaction_Amount'
    mu_amount_s2 = fraud_data['Transaction_Amount'].mean()
    sigma_amount_s2 = fraud_data['Transaction_Amount'].std()
    mu_amount_s1 = non_fraud_data['Transaction_Amount'].mean()
    sigma_amount_s1 = non_fraud_data['Transaction_Amount'].std()

    # Fit a Weibull distribution to 'Time_of_Transaction'
    shape_s2, loc_s2, scale_s2 = weibull_min.fit(fraud_data['Time_of_Transaction'], floc=0)
    shape_s1, loc_s1, scale_s1 = weibull_min.fit(non_fraud_data['Time_of_Transaction'], floc=0)
    
    return {
        'mu_amount_s1': mu_amount_s1,
        'sigma_amount_s1': sigma_amount_s1,
        'shape_s1': shape_s1,
        'scale_s1': scale_s1,  # Weibull parameters for non-fraudulent time
        'mu_amount_s2': mu_amount_s2,
        'sigma_amount_s2': sigma_amount_s2,
        'shape_s2': shape_s2,
        'scale_s2': scale_s2   # Weibull parameters for fraudulent time
    }

def gaussian_probability(x, mu, sigma):
    """
    Calculate the Gaussian probability of x given mean (mu) and standard deviation (sigma).
    """
    return norm.pdf(x, loc=mu, scale=sigma)

def weibull_probability(x, shape, scale):
    """
    Calculate the Weibull probability of x given shape and scale parameters.
    """
    return weibull_min.pdf(x, c=shape, scale=scale)

def calculate_state_1_prob(transaction, mu_amount_s1, sigma_amount_s1, shape_s1, scale_s1, mu_distance_s1, sigma_distance_s1):
    prob_amount = gaussian_probability(transaction["amount"], mu_amount_s1, sigma_amount_s1)
    prob_time = weibull_probability(transaction["time"], shape_s1, scale_s1)
    prob_distance = gaussian_probability(transaction["distance"], mu_distance_s1, sigma_distance_s1)
    return prob_amount * prob_time * prob_distance

def calculate_state_2_prob(transaction, mu_amount_s2, sigma_amount_s2, shape_s2, scale_s2, mu_distance_s2, sigma_distance_s2):
    prob_amount = gaussian_probability(transaction["amount"], mu_amount_s2, sigma_amount_s2)
    prob_time = weibull_probability(transaction["time"], shape_s2, scale_s2)
    prob_distance = gaussian_probability(transaction["distance"], mu_distance_s2, sigma_distance_s2)
    return prob_amount * prob_time * prob_distance

# -----------------------------------------
# New Function: 3D Correlation Plot
# -----------------------------------------
def plot_3d_correlation(mu, sigma, shape, scale, x_min=0.01, x_max=5, num_points=200):
    """
    Plots a 3D correlation scatter plot between the Normal PDF and Weibull PDF.

    Parameters:
        mu (float): Mean of the Normal distribution.
        sigma (float): Standard deviation of the Normal distribution.
        shape (float): Shape parameter (c) of the Weibull distribution.
        scale (float): Scale parameter of the Weibull distribution.
        x_min (float): Minimum x value (default is 0.01 to avoid issues with Weibull at 0).
        x_max (float): Maximum x value.
        num_points (int): Number of points in the x range.

    Returns:
        fig (plotly.graph_objs.Figure): The 3D scatter plot figure.
    """
    # Create an array of x values.
    x = np.linspace(x_min, x_max, num_points)
    
    # Compute the PDFs for both distributions.
    normal_pdf = norm.pdf(x, loc=mu, scale=sigma)
    weibull_pdf = weibull_min.pdf(x, c=shape, scale=scale)
    
    # Create a 3D scatter plot.
    trace = go.Scatter3d(
        x=x,
        y=normal_pdf,
        z=weibull_pdf,
        mode='markers',
        marker=dict(
            size=5,
            color=weibull_pdf,  # Color by Weibull PDF values
            colorscale='Viridis',
            opacity=0.8
        )
    )
    
    layout = go.Layout(
        title='3D Correlation Plot: Normal PDF vs. Weibull PDF',
        scene=dict(
            xaxis_title='x',
            yaxis_title='Normal PDF',
            zaxis_title='Weibull PDF'
        )
    )
    
    fig = go.Figure(data=[trace], layout=layout)
    
    # Compute and print the Pearson correlation coefficient.
    correlation_matrix = np.corrcoef(normal_pdf, weibull_pdf)
    pearson_corr = correlation_matrix[0, 1]
    print(f"Pearson correlation coefficient: {pearson_corr:.4f}")
    
    return fig