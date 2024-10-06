import pandas as pd
import numpy as np
import os

# Define the directory containing your CSV files
data_dir = '/content/training_dataset'
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

sampling_rate = 120
sta_window_sec = 50
lta_window_sec = 610
n_sta = sta_window_sec * sampling_rate
n_lta = lta_window_sec * sampling_rate

def compute_sta_lta(data, n_sta, n_lta):
    sta = np.convolve(data**2, np.ones(n_sta)/n_sta, mode='same')  
    lta = np.convolve(data**2, np.ones(n_lta)/n_lta, mode='same')  
    with np.errstate(divide='ignore', invalid='ignore'):
        sta_lta = np.where(lta > 0, sta / lta, 0)  
    return sta_lta

def group_events(event_indices, sta_lta_ratio, min_gap):
    grouped_events = []
    current_group = [event_indices[0]]
    
    for i in range(1, len(event_indices)):
        if event_indices[i] - event_indices[i - 1] <= min_gap:
            current_group.append(event_indices[i])
        else:
            max_index = current_group[np.argmax(sta_lta_ratio[current_group])]
            grouped_events.append(max_index)
            current_group = [event_indices[i]]
    
    if current_group:
        max_index = current_group[np.argmax(sta_lta_ratio[current_group])]
        grouped_events.append(max_index)
    
    return grouped_events

min_gap = 5 * sampling_rate

# Process each CSV file
all_results = []  # To store results from all files
for csv_file in csv_files:
    file_path = os.path.join(data_dir, csv_file)
    df = pd.read_csv(file_path)
    df = df.dropna()
    df['time_abs'] = pd.to_datetime(df['time_abs(%Y-%m-%dT%H:%M:%S.%f)'])
    
    velocity = df['velocity(m/s)'].values
    sta_lta_ratio = compute_sta_lta(velocity, n_sta, n_lta)
    
    # Dynamically compute threshold based on data characteristics
    mean_sta_lta = np.mean(sta_lta_ratio)
    std_sta_lta = np.std(sta_lta_ratio)
    k = 5  # Adjust this multiplier to change sensitivity
    threshold = mean_sta_lta + k * std_sta_lta
    print(f"Threshold for {csv_file}: {threshold}")
    
    event_indices = np.where(sta_lta_ratio > threshold)[0]
    grouped_event_indices = group_events(event_indices, sta_lta_ratio, min_gap)

    # Check if any events were detected
    if grouped_event_indices:  # Only proceed if there are detected events
        # Output detected events (time of occurrence at peak points)
        event_times = df['time_abs'][grouped_event_indices]
        relative_times = df['time_rel(sec)'][grouped_event_indices]

        # Create a DataFrame to store the results
        results_df = pd.DataFrame({
            'Absolute Time': event_times,
            'Relative Time (sec)': relative_times,
            'File': csv_file  # Keep track of the file name
        })

        all_results.append(results_df)
    else:
        print(f"No seismic events detected in {csv_file}, skipping...")

# Concatenate results from all files into a single DataFrame
if all_results:
    final_results = pd.concat(all_results, ignore_index=True)
    
    # Save the results to a CSV file
    output_file_path = '/content/detected_event.csv'
    final_results.to_csv(output_file_path, index=False)

    # Print confirmation message
    print(f"Detected seismic events saved to {output_file_path}")
else:
    print("No seismic events detected in any files.")
