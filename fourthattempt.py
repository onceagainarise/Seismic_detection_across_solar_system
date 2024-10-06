import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('C:/nasaapp/space_apps_2024_seismic_detection/space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA/xa.s12.00.mhz.1970-04-25HR00_evid00006.csv')
df=df.dropna()


df['time_abs'] = pd.to_datetime(df['time_abs(%Y-%m-%dT%H:%M:%S.%f)'])


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


velocity = df['velocity(m/s)'].values
sta_lta_ratio = compute_sta_lta(velocity, n_sta, n_lta)


threshold = 4.2 
event_indices = np.where(sta_lta_ratio > threshold)[0]

#
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


grouped_event_indices = group_events(event_indices, sta_lta_ratio, min_gap)


plt.figure(figsize=(12, 6))


plt.subplot(2, 1, 1)
plt.plot(df['time_rel(sec)'], velocity, label="Velocity (m/s)")
plt.scatter(df['time_rel(sec)'][grouped_event_indices], velocity[grouped_event_indices], color='red', label="Detected Peak", marker='x')
plt.xlabel('Time (sec)')
plt.ylabel('Velocity (m/s)')
plt.title('Seismic Event Detection with Peaks')
plt.legend()


plt.subplot(2, 1, 2)
plt.plot(df['time_rel(sec)'], sta_lta_ratio, label="STA/LTA Ratio")
plt.axhline(y=threshold, color='red', linestyle='--', label="Threshold")
plt.scatter(df['time_rel(sec)'][grouped_event_indices], sta_lta_ratio[grouped_event_indices], color='red', label="Detected Peak", marker='x')
plt.xlabel('Time (sec)')
plt.ylabel('STA/LTA Ratio')
plt.legend()

plt.tight_layout()
plt.show()


# Output detected events (time of occurrence at peak points)
event_times = df['time_abs'][grouped_event_indices]
relative_times = df['time_rel(sec)'][grouped_event_indices]

# Create a DataFrame to store the results
results_df = pd.DataFrame({
    'Absolute Time': event_times,
    'Relative Time (sec)': relative_times
})

# Print the results DataFrame
print("Detected seismic events:\n", results_df)
