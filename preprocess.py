import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(input_data):
    # # Drop 'Class' column
    # input_data.drop('Class', axis=1, inplace=True)

    # Drop columns with constant values
    for col in input_data.columns:
        if np.min(input_data[col]) == np.max(input_data[col]):
            input_data.drop(col, axis=1, inplace=True)

    # Scale features using MinMax scaler
    scaler = MinMaxScaler()
    input_data_scaled = scaler.fit_transform(input_data)
    input_data_scaled = pd.DataFrame(input_data_scaled, columns=input_data.columns)

    # Reshape input data to LSTM format [samples, time_steps, features]
    input_data_lstm = input_data_scaled.values.reshape(input_data_scaled.shape[0], 1, input_data_scaled.shape[1])

    # # Create a DataFrame with reshaped data and column names
    # input_data_lstm_df = pd.DataFrame(input_data_lstm[:, 0, :], columns=input_data.columns)
    
    return input_data_lstm
