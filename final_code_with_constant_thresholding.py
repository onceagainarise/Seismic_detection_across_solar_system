import pandas as pd
import numpy as np
import os

# Define the directory containing your CSV files
data_dir = 'training_dataset'  # Directory path
# List all CSV files in the specified directory
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

# Define sampling parameters
sampling_rate = 120  # Sampling rate in Hz
sta_window_sec = 50  # Short-term average window in seconds
lta_window_sec = 610  # Long-term average window in seconds

# Calculate the number of samples for STA and LTA windows
n_sta = sta_window_sec * sampling_rate
n_lta = lta_window_sec * sampling_rate

def compute_sta_lta(data, n_sta, n_lta):
    """
    Compute the Short-Term Average (STA) and Long-Term Average (LTA) 
    of the squared input data using convolution.
    
    Parameters:
    data (ndarray): Input velocity data.
    n_sta (int): Number of samples for the STA window.
    n_lta (int): Number of samples for the LTA window.
    
    Returns:
    ndarray: STA/LTA ratio.
    """
    # Compute STA and LTA using moving averages
    sta = np.convolve(data**2, np.ones(n_sta) / n_sta, mode='same')
    lta = np.convolve(data**2, np.ones(n_lta) / n_lta, mode='same')
    # Compute the STA/LTA ratio, avoiding division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        sta_lta = np.where(lta > 0, sta / lta, 0)
    return sta_lta

def group_events(event_indices, sta_lta_ratio, min_gap):
    """
    Group detected event indices based on a minimum gap.
    
    Parameters:
    event_indices (ndarray): Indices where events are detected.
    sta_lta_ratio (ndarray): STA/LTA ratio.
    min_gap (int): Minimum gap (in samples) to consider events separate.
    
    Returns:
    list: Indices of grouped events.
    """
    grouped_events = []  # To store final grouped event indices
    current_group = [event_indices[0]]  # Start with the first event

    # Iterate through detected events
    for i in range(1, len(event_indices)):
        # If the gap between events is less than or equal to min_gap
        if event_indices[i] - event_indices[i - 1] <= min_gap:
            current_group.append(event_indices[i])  # Add to current group
        else:
            # Get the maximum STA/LTA index from the current group
            max_index = current_group[np.argmax(sta_lta_ratio[current_group])]
            grouped_events.append(max_index)  # Store it
            current_group = [event_indices[i]]  # Start a new group

    # Handle the last group if it exists
    if current_group:
        max_index = current_group[np.argmax(sta_lta_ratio[current_group])]
        grouped_events.append(max_index)

    return grouped_events

# Define minimum gap between events in samples (5 seconds)
min_gap = 5 * sampling_rate

# Process each CSV file
all_results = []  # List to store results from all files
for csv_file in csv_files:
    file_path = os.path.join(data_dir, csv_file)  # Construct the full file path
    df = pd.read_csv(file_path)  # Read the CSV file
    df = df.dropna()  # Remove any rows with missing values
    df['time_abs'] = pd.to_datetime(df['time_abs(%Y-%m-%dT%H:%M:%S.%f)'])  # Convert time to datetime format

    velocity = df['velocity(m/s)'].values  # Extract velocity data
    sta_lta_ratio = compute_sta_lta(velocity, n_sta, n_lta)  # Compute STA/LTA ratio

    threshold = 4.2  # Define threshold for event detection
    # Get indices of events where STA/LTA ratio exceeds the threshold
    event_indices = np.where(sta_lta_ratio > threshold)[0]
    # Group the detected events based on minimum gap
    grouped_event_indices = group_events(event_indices, sta_lta_ratio, min_gap)

    # Output detected events (time of occurrence at peak points)
    event_times = df['time_abs'][grouped_event_indices]  # Absolute times of events
    relative_times = df['time_rel(sec)'][grouped_event_indices]  # Relative times of events

    # Create a DataFrame to store the results for the current file
    results_df = pd.DataFrame({
        'Absolute Time': event_times,
        'Relative Time (sec)': relative_times,
        'File': csv_file  # Keep track of the file name
    })

    all_results.append(results_df)  # Append current results to the list

# Concatenate results from all files into a single DataFrame
final_results = pd.concat(all_results, ignore_index=True)

# Save the results to a CSV file
output_file_path = 'write_the_path_of_the_file_where_you_want_to_store_the_sesmic_data.csv'  # Output file path
final_results.to_csv(output_file_path, index=False)  # Save results

# Print confirmation message
print(f"Detected seismic events saved to {output_file_path}")  # Confirmation of output
