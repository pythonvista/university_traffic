import streamlit as st
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the trained LSTM model and scaler
# @st.cache_data(allow_output_mutation=True)
def load_model_and_scaler():
    model = load_model('./models/best_model_weights.keras')  # Update with correct model path
    with open('./models/minmax_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

# Preprocess the data similar to your notebook preprocessing
def preprocess2(df):
    scaler2 = MinMaxScaler()
    df.index = pd.to_datetime(df.Datetime)
    new_df = df[['Average_Receive_bps', 'Average_Transmit_bps']]
    scaled_data = scaler2.fit_transform(new_df)
    sequence_length = 10  # Number of time steps in each sequence
    num_features = len(new_df.columns)
    # Create sequences and corresponding labels
    sequences = []
    labels = []
    for i in range(len(scaled_data) - sequence_length):
        seq = scaled_data[i:i+sequence_length]
        label = scaled_data[i+sequence_length][1]  # 'average_recieve_bmp' column index
        sequences.append(seq)
        labels.append(label)

    # Convert to numpy arrays
    sequences = np.array(sequences)
    labels = np.array(labels)

    # Split into train and test sets
    train_size = int(0.8 * len(sequences))
    train_x, test_x = sequences[:train_size], sequences[train_size:]
    train_y, test_y = labels[:train_size], labels[train_size:]
    return test_x, test_y,scaler2,new_df

def Predictor(test_x,test_y,scaler, model, new_df):
    # y_true values
    test_y_copies = np.repeat(test_y.reshape(-1, 1), test_x.shape[-1], axis=-1)
    true_temp = scaler.inverse_transform(test_y_copies)[:,1]

    # predicted values
    prediction = model.predict(test_x)
    prediction_copies = np.repeat(prediction, 2, axis=-1)
    predicted_temp = scaler.inverse_transform(prediction_copies)[:,1]
    # Plotting predicted and actual values
    st.write("Traffic Forecasting Plot:")
    plt.figure(figsize=(12, 7))  # Increase the size for better readability
    plt.plot(new_df.index[-100:], true_temp[-100:], color='blue', linestyle='-', marker='', label='Actual')
    plt.plot(new_df.index[-100:], predicted_temp[-100:], color='red', linestyle='--', marker='', label='Predicted')

    plt.title('Average Receive Bps Prediction vs Actual', fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Average Receive Bps', fontsize=12)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True)  # Add a grid for better readability
    plt.tight_layout() 
    st.pyplot(plt) # Adjust layout to fit labels and titles
    # plt.show()

def preprocess_data(df, scaler):
    # Assuming 'Average_Transmit_bps' and 'Average_Receive_bps' are the input columns used in your model
    n_steps = 10
    
    # Convert 'Datetime' to pandas datetime type and sort by it
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    
    # Sort the DataFrame by 'Datetime' in ascending order
    df_sorted = df.sort_values(by='Datetime')
    
    # Select the relevant columns for scaling (you want only the last 10 rows)
    df_scaled = df_sorted[['Average_Receive_bps', 'Average_Transmit_bps']].tail(n_steps)
    original = df_scaled
    # Scale the last n_steps data (just the last 10 rows)
    df_scaled = scaler.transform(df_scaled)
    
    # Reshape the data to be compatible with LSTM input shape (samples, timesteps, features)
    X_new = df_scaled.reshape((1, n_steps, df_scaled.shape[1]))  # Now it will have shape (1, 10, 2)
    
    return X_new ,original



# Predict the traffic using the LSTM model
def make_predictions(model, data):
    predictions = model.predict(data)
    return predictions

# Denormalize the predictions
def inverse_transform(predictions, scaler):
    # Assuming the second column ('Average_Receive_bps') is the target variable
    prediction_copies = np.repeat(predictions, 2, axis=-1)
    print(prediction_copies)
    predicted_temp = scaler.inverse_transform(prediction_copies)[:,0]
    return predicted_temp

# Main Streamlit app
def main():
    st.title("University Traffic Forecasting with LSTM")
    
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    
    # File uploader to upload traffic data
    uploaded_file = st.file_uploader("Upload your traffic data CSV", type=["csv"])
    
    if uploaded_file is not None:
        # Read the uploaded CSV file
        df = pd.read_csv(uploaded_file)
        
        # Display the uploaded data
        st.write("Uploaded Data:")
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        st.dataframe(df.head())
        n_steps = 10
        # Preprocess the data
        test_x, test_y,scaler2, new_df = preprocess2(df)
        print(test_x)
        Predictor(test_x, test_y, scaler2, model,new_df)
        # data_scaled, original = preprocess_data(df, scaler)
        
        # # Make predictions
        # predictions = make_predictions(model, data_scaled)
        # print(predictions)
        
        # # # # Denormalize predictions
        # traffic_forecast = inverse_transform(predictions, scaler)
        # print(traffic_forecast)
        
        # df_scaled_last_n_steps['Predicted_Average_Receive_bps'] = traffic_forecast
        # st.write("Predicted Traffic:")
        # st.dataframe(df_scaled_last_n_steps[[ 'Average_Receive_bps', 'Average_Transmit_bps', 'Predicted_Average_Receive_bps']])
        
        # # Plotting the results
        # st.write("Traffic Forecasting Plot:")
        # plt.figure(figsize=(10, 6))
        # plt.plot(df_scaled_last_n_steps['Predicted_Average_Receive_bps'], label='Predicted Traffic')
        # plt.xlabel('Time')
        # plt.ylabel('Traffic (bps)')
        # plt.legend()
        # st.pyplot(plt)

if __name__ == "__main__":
    main()
