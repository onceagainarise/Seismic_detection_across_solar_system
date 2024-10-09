import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

data_dir = '/content/mars_dataset'
plot_dir = '/content/mars_dataset_plot'

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

sampling_rate = 50
sta_window_sec = 30
lta_window_sec = 500
n_sta = int(sta_window_sec * sampling_rate)
n_lta = int(lta_window_sec * sampling_rate)

def compute_sta_lta(data, n_sta, n_lta):
    sta = np.convolve(data**2, np.ones(n_sta)/n_sta, mode='same')
    lta = np.convolve(data**2, np.ones(n_lta)/n_lta, mode='same')
    with np.errstate(divide='ignore', invalid='ignore'):
        sta_lta = np.where(lta > 0, sta / lta, 0)
    return sta_lta

def group_events(event_indices, sta_lta_ratio, min_gap):
    if len(event_indices) == 0:
        return []

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
all_peak_results = []

for csv_file in csv_files:
    file_path = os.path.join(data_dir, csv_file)

    df = pd.read_csv(file_path, dtype={'velocity(c/s)': str}, low_memory=False)
    df = df.dropna()
    df['time_abs'] = pd.to_datetime(df['time(%Y-%m-%dT%H:%M:%S.%f)'])

    df['velocity(c/s)'] = pd.to_numeric(df['velocity(c/s)'], errors='coerce')
    df = df.dropna(subset=['velocity(c/s)'])

    if df.empty:
        print(f"No valid velocity data in {csv_file}, skipping this file.")
        continue

    velocity = df['velocity(c/s)'].values
    sta_lta_ratio = compute_sta_lta(velocity, n_sta, n_lta)

    mean_sta_lta = np.mean(sta_lta_ratio)
    std_sta_lta = np.std(sta_lta_ratio)
    k = 2.8
    threshold = mean_sta_lta + k * std_sta_lta
    print(f"Threshold for {csv_file}: {threshold}")

    event_indices = np.where(sta_lta_ratio > threshold)[0]

    if len(event_indices) == 0:
        print(f"No seismic events detected in {csv_file}, skipping this file.")
        continue

    grouped_event_indices = group_events(event_indices, sta_lta_ratio, min_gap)

    if grouped_event_indices:
        peak_index = grouped_event_indices[np.argmax(sta_lta_ratio[grouped_event_indices])]
        peak_value = sta_lta_ratio[peak_index]
        peak_time_abs = df['time_abs'].iloc[peak_index]
        peak_time_rel = df['rel_time(sec)'].iloc[peak_index]

        all_peak_results.append({
            'Peak STA/LTA Value': peak_value,
            'Absolute Time': peak_time_abs,
            'Relative Time (sec)': peak_time_rel,
            'File': csv_file
        })

        plt.figure(figsize=(10, 6))
        plt.plot(df['time_abs'], sta_lta_ratio, label='STA/LTA Ratio', color='b')
        plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.2f}')
        plt.scatter(df['time_abs'].iloc[grouped_event_indices],
                    sta_lta_ratio[grouped_event_indices],
                    color='orange', label='Detected Events')

        plt.title(f'STA/LTA Detection for {csv_file} (Mars Data)')
        plt.xlabel('Time')
        plt.ylabel('STA/LTA Ratio')
        plt.legend()

        plot_output_file = os.path.join(plot_dir, f'{csv_file}_sta_lta_plot.png')
        plt.savefig(plot_output_file)
        plt.close()

        print(f"Plot saved for {csv_file} at {plot_output_file}")
    else:
        print(f"No seismic events detected in {csv_file}, skipping...")

if all_peak_results:
    peak_results_df = pd.DataFrame(all_peak_results)
    output_file_path = '/content/event_found.csv'
    peak_results_df.to_csv(output_file_path, index=False)

    print(f"Detected peak seismic events saved to {output_file_path}")
else:
    print("No seismic events detected in any files.")
