import streamlit as st
import tensorflow as tf
import pandas as pd 
import joblib
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
import sys
import codecs


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Load the trained LSTM model and scaler
# @st.cache_data(allow_output_mutation=True)
def load_model_and_scaler():
    tf.random.set_seed(42)
    try:
        model = load_model('./models/best_model_weights.keras')  # Update with correct model path
        scaler = joblib.load('./models/minmax_scaler.pkl')
        return model, scaler
    except Exception as e:
        print(f"Error loading model: {e}")
    

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


    st.title("University Traffic Forecasting with LSTM")
    
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    
    # Add inputs for manual data entry of previous values (Datetime, Average_Receive_bps, Average_Transmit_bps)
    st.write("Enter the last 4 data points for prediction:")
    
    dates = []
    avg_receive_bps = []
    avg_transmit_bps = []

    for i in range(10):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            date = st.date_input(f"Date {i+1}", key=f"date_{i}")
            time = st.time_input(f"Time {i+1}", key=f"time_{i}")
            datetime_input = pd.to_datetime(f"{date} {time}")
            dates.append(datetime_input)
        
        with col2:
            receive_bps = st.number_input(f"Average_Receive_bps {i+1}", key=f"receive_bps_{i}")
            avg_receive_bps.append(receive_bps)
        
        with col3:
            transmit_bps = st.number_input(f"Average_Transmit_bps {i+1}", key=f"transmit_bps_{i}")
            avg_transmit_bps.append(transmit_bps)

    # Create a DataFrame from the input values
    if st.button("Predict"):
        model, scaler = load_model_and_scaler()
        df_input = pd.DataFrame({
            'Datetime': dates,
            'Average_Receive_bps': avg_receive_bps,
            'Average_Transmit_bps': avg_transmit_bps
        })

        # Display the input data
        st.write("Input Data for Prediction:")
        st.dataframe(df_input)

        # # Preprocess the input data
        X_new, original_data = preprocess_data(df_input, scaler) 
      

        # Make predictions
        predictions = make_predictions(model, X_new)

        # Inverse transform to get the actual values
        predicted_value = inverse_transform(predictions, scaler)

        # Display the predicted result
        st.write(f"Predicted Average_Receive_bps: {predicted_value[0]}")

        # Plot the data
        plt.figure(figsize=(8, 4))
        plt.plot(original_data['Datetime'], original_data['Average_Receive_bps'], label="Original Data", color="blue")
        plt.scatter(original_data['Datetime'].iloc[-1], predicted_value[0], color="red", label="Predicted Value")
        plt.title("Traffic Forecasting")
        plt.xlabel("Datetime")
        plt.ylabel("Average_Receive_bps")
        plt.legend()
        st.pyplot(plt)


def Forecast(scaler, model):
    
    st.write("Enter the last 10 data points for prediction:")
    uploaded_forcast = st.file_uploader("Upload up to 10 recent traffic", type=["csv"])
    
    if uploaded_forcast is not None:
        st.write("Predicting:")
        try:
            # Load the CSV data
            df = pd.read_csv(uploaded_forcast)
            st.write("Uploaded Data:")

            # Ensure 'Datetime' column is in datetime format
            df['Datetime'] = pd.to_datetime(df['Datetime'])

            # Check if the dataframe has at least 10 rows
            if len(df) >= 10:
                # Pick the last 10 rows from the dataframe
                last_10_rows = df[['Average_Receive_bps', 'Average_Transmit_bps']].tail(10)
                st.write("Last 10 rows of the data to be used for forecasting:")
                st.dataframe(last_10_rows)
                df_scaled = scaler.transform(last_10_rows[['Average_Receive_bps', 'Average_Transmit_bps']])

                # Reshape the data to be compatible with LSTM input shape (samples, timesteps, features)
                n_steps = 10  # The sequence length used during training
                X_new = df_scaled.reshape((1, n_steps, df_scaled.shape[1])) 
                predicted_scaled_value = model.predict(X_new, verbose=0)
                mean_average_transmit_bps = np.mean(X_new[0][:, 1])  # Taking the mean of the second column

                # Create a dummy array with the predicted value and the calculated mean
                dummy_array = np.array([[predicted_scaled_value[0][0], mean_average_transmit_bps]])

                # Inverse transform the predictions
                inverse_scaled_value = scaler.inverse_transform(dummy_array)

                # Extract the predicted 'Average_Receive_bps' (assuming it is the first feature)
                Predicted_Average_Receive_bp = inverse_scaled_value[0][0]  # This retrieves the first feature from the inverse transformation
                Predicted_Average_Transmit_bp = inverse_scaled_value[0][1]

                st.write(f"Predicted Average_Receive_bps: {Predicted_Average_Receive_bp}")
                st.write(f"Predicted Average_Transmit_bps: {Predicted_Average_Transmit_bp}")
            else:
                st.warning("We need at least 10 previous data points to forecast.")

        except FileNotFoundError:
            st.error("File not found. Please upload the data.")
    else:
        st.write("Please upload a file to proceed with forecasting.")



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

    
    Forecast(scaler, model)

if __name__ == "__main__":
    main()
