import pandas as pd
import numpy as np
import os

# Define the directory containing your CSV files
data_dir = 'path_to_your_directory/folder'
# List all CSV files in the specified directory
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

# Set sampling rate and STA/LTA window lengths (in seconds)
sampling_rate = 120
sta_window_sec = 50  # Short-term average window
lta_window_sec = 610  # Long-term average window

# Calculate the number of samples for STA and LTA based on the sampling rate
n_sta = sta_window_sec * sampling_rate
n_lta = lta_window_sec * sampling_rate

def compute_sta_lta(data, n_sta, n_lta):
    """
    Calculate the STA/LTA ratio for a given data array.
    
    Parameters:
        data (numpy array): The input seismic velocity data.
        n_sta (int): Number of samples for the short-term average (STA).
        n_lta (int): Number of samples for the long-term average (LTA).
        
    Returns:
        numpy array: The STA/LTA ratio.
    """
    # Compute the short-term average (STA) using convolution
    sta = np.convolve(data**2, np.ones(n_sta)/n_sta, mode='same')  
    # Compute the long-term average (LTA) using convolution
    lta = np.convolve(data**2, np.ones(n_lta)/n_lta, mode='same')  
    
    # Calculate the STA/LTA ratio; handle division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        sta_lta = np.where(lta > 0, sta / lta, 0)  
        
    return sta_lta

def group_events(event_indices, sta_lta_ratio, min_gap):
    """
    Group detected events based on indices and a minimum gap threshold.
    
    Parameters:
        event_indices (list): Indices of detected events.
        sta_lta_ratio (numpy array): The STA/LTA ratio.
        min_gap (int): Minimum gap in indices to consider events as part of the same group.
        
    Returns:
        list: Indices of grouped events.
    """
    grouped_events = []  # List to store indices of grouped events
    current_group = [event_indices[0]]  # Start the first group with the first event
    
    # Iterate through the detected event indices
    for i in range(1, len(event_indices)):
        # If the current event is within the minimum gap from the previous, add to the group
        if event_indices[i] - event_indices[i - 1] <= min_gap:
            current_group.append(event_indices[i])
        else:
            # If the gap is too large, save the index of the peak event in the current group
            max_index = current_group[np.argmax(sta_lta_ratio[current_group])]
            grouped_events.append(max_index)  # Append the index of the peak event
            current_group = [event_indices[i]]  # Start a new group
    
    # Handle any remaining events in the last group
    if current_group:
        max_index = current_group[np.argmax(sta_lta_ratio[current_group])]
        grouped_events.append(max_index)
    
    return grouped_events

# Define the minimum gap (in samples) between events for grouping
min_gap = 5 * sampling_rate

# Process each CSV file in the directory
all_results = []  # To store results from all files
for csv_file in csv_files:
    # Construct the full file path
    file_path = os.path.join(data_dir, csv_file)
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    # Drop any rows with missing values
    df = df.dropna()
    # Convert absolute time to datetime format
    df['time_abs'] = pd.to_datetime(df['time_abs(%Y-%m-%dT%H:%M:%S.%f)'])
    
    # Extract the velocity data as a NumPy array
    velocity = df['velocity(m/s)'].values
    # Compute the STA/LTA ratio for the velocity data
    sta_lta_ratio = compute_sta_lta(velocity, n_sta, n_lta)
    
    # Dynamically compute a threshold based on the STA/LTA ratio's mean and standard deviation
    mean_sta_lta = np.mean(sta_lta_ratio)
    std_sta_lta = np.std(sta_lta_ratio)
    k = 6  # Adjust this multiplier to change sensitivity (higher values = more sensitivity)
    threshold = mean_sta_lta + k * std_sta_lta
    print(threshold)  # Print the computed threshold for reference
    
    # Identify indices where the STA/LTA ratio exceeds the threshold
    event_indices = np.where(sta_lta_ratio > threshold)[0]
    # Group detected events based on the identified indices
    grouped_event_indices = group_events(event_indices, sta_lta_ratio, min_gap)

    # Output detected events (time of occurrence at peak points)
    event_times = df['time_abs'][grouped_event_indices]  # Absolute times of events
    relative_times = df['time_rel(sec)'][grouped_event_indices]  # Relative times of events

    # Create a DataFrame to store the results for this file
    results_df = pd.DataFrame({
        'Absolute Time': event_times,
        'Relative Time (sec)': relative_times,
        'File': csv_file  # Keep track of the file name from which events were detected
    })

    all_results.append(results_df)  # Append results for this file to the list

# Concatenate results from all files into a single DataFrame
final_results = pd.concat(all_results, ignore_index=True)

# Save the final results to a CSV file
output_file_path = 'write_the_path_of_the_file_where_you_want_to_store_the_sesmic_data.csv'
final_results.to_csv(output_file_path, index=False)

# Print a confirmation message indicating where the results were saved
print(f"Detected seismic events saved to {output_file_path}")
