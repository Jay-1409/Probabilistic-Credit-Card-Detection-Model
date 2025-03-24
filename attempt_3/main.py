import numpy as np
import pandas as pd
import plotly.graph_objs as go
from scipy.stats import norm, weibull_min
from calculations import (calculate_variable_values, 
                          calculate_state_1_prob, 
                          calculate_state_2_prob, 
                          gaussian_probability, 
                          weibull_probability)

# Original Transition probabilities (A)
a = np.random.rand()
b = np.random.rand()
c = np.random.rand()
d = np.random.rand()
A = np.array([[a, b],   # Transition from Legitimate (S1)
              [c, d]])  # Transition from Fraudulent (S2)

# Initial state probabilities (pi)
pi = np.array([1, 0])  # In this example, we assume 100% chance starting in Legitimate

def plot_3d_correlation(mu, sigma, shape, scale, x_min=0.01, x_max=5, num_points=200):
    """
    Plots a 3D scatter plot comparing the Normal PDF and the Weibull PDF.
    
    Parameters:
      mu (float): Mean of the Normal distribution.
      sigma (float): Standard deviation of the Normal distribution.
      shape (float): Shape parameter (c) for the Weibull distribution.
      scale (float): Scale parameter for the Weibull distribution.
      x_min (float): Minimum x value (default 0.01).
      x_max (float): Maximum x value.
      num_points (int): Number of points in x range.
      
    Returns:
      fig: A Plotly figure object.
    """
    x = np.linspace(x_min, x_max, num_points)
    normal_pdf = norm.pdf(x, loc=mu, scale=sigma)
    weibull_pdf = weibull_min.pdf(x, c=shape, scale=scale)
    
    trace = go.Scatter3d(
        x=x,
        y=normal_pdf,
        z=weibull_pdf,
        mode='markers',
        marker=dict(
            size=5,
            color=weibull_pdf,
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
    # Compute and print the Pearson correlation coefficient
    correlation_matrix = np.corrcoef(normal_pdf, weibull_pdf)
    pearson_corr = correlation_matrix[0, 1]
    # print(f"Pearson correlation coefficient: {pearson_corr:.4f}")
    return fig

def verdict(prev_amt, prev_time, amt, time, plot_pdf=True):
    """
    Calculates the probabilities for two transactions and returns a verdict.
    Optionally, it can generate a 3D correlation plot between a Normal and Weibull PDF
    using the parameters from the legitimate state.
    
    Parameters:
      prev_amt (float): Amount of the previous transaction.
      prev_time (float): Time of the previous transaction.
      amt (float): Amount of the current transaction.
      time (float): Time of the current transaction.
      plot_pdf (bool): If True, display the 3D correlation plot.
      
    Returns:
      bool: True if current transaction is classified as fraudulent, else False.
    """
    # Get parameters from data
    variables = calculate_variable_values()
    
    # Parameters for State 1 (Legitimate)
    mu_amount_s1, sigma_amount_s1 = variables['mu_amount_s1'], variables['sigma_amount_s1']
    shape_s1, scale_s1 = variables['shape_s1'], variables['scale_s1']
    mu_distance_s1, sigma_distance_s1 = 10, 5

    # Parameters for State 2 (Fraudulent)
    mu_amount_s2, sigma_amount_s2 = variables['mu_amount_s2'], variables['sigma_amount_s2']
    shape_s2, scale_s2 = variables['shape_s2'], variables['scale_s2']
    mu_distance_s2, sigma_distance_s2 = 100, 50
    
    # Define two transactions
    transaction_1 = {"amount": prev_amt, "time": prev_time, "distance": 120}
    transaction_2 = {"amount": amt, "time": time, "distance": 1}
    
    # Compute state probabilities for Transaction 1
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
    
    # Compute state probabilities for Transaction 2
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
    
    # Compute total probability for Transaction 2 considering state transitions
    total_prob_transaction_2 = (
        p_s1_transaction_1 * A[0, 0] * p_s1_transaction_2 +
        p_s1_transaction_1 * A[0, 1] * p_s2_transaction_2 +
        p_s2_transaction_1 * A[1, 0] * p_s1_transaction_2 +
        p_s2_transaction_1 * A[1, 1] * p_s2_transaction_2
    )
    
    # Use a threshold to decide if transaction is fraudulent.
    fraud_threshold = 0.5
    
    def is_fraudulent(p_s1, p_s2, threshold):
        total = p_s1 + p_s2
        prob_s1 = p_s1 / total
        prob_s2 = p_s2 / total
        return True if prob_s2 > threshold else False
    
    decision = is_fraudulent(p_s1_transaction_2, p_s2_transaction_2, fraud_threshold)
    
    # Optionally, generate and show a 3D correlation plot
    if plot_pdf:
        # For demonstration, we use State 1 parameters to plot a Normal PDF vs. Weibull PDF.
        # You can choose any parameters here. We're using mu_amount_s1, sigma_amount_s1 and shape_s1, scale_s1.
        fig = plot_3d_correlation(mu_amount_s1, sigma_amount_s1, shape_s1, scale_s1)
        fig.write_html("plot.html")
        # print("Plot saved as 'plot.html'. Please open this file in your browser to view the graph.")
    return decision