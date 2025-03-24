from sklearn.preprocessing import StandardScaler
def scale_data(data):    
    scaler = StandardScaler()
    new_data = scaler.fit_transform(data[['Transaction_Amount', 'Time_of_Transaction']])
    
    return new_data