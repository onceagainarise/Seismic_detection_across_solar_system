# Seismic Event Detection in Planetary Seismology Missions

## Overview

This project addresses the challenge of identifying seismic events within noisy planetary seismic data, such as that collected by the Apollo missions and the Mars InSight Lander. Transmitting continuous seismic data from a distant lander to Earth is power-intensive, and not all of the data is scientifically useful. Therefore, this project focuses on creating a solution that can distinguish meaningful seismic events from noise and only transmit relevant data back to Earth.

## Objective

The goal is to create an algorithm capable of identifying seismic events within the data. The algorithm is designed to process planetary seismic data from missions like Apollo and Mars InSight, filtering out noise, detecting seismic quakes, and highlighting significant events.

### Key Features:
- **Noise filtering**: Distinguishes signal from noise using the Short-Term Average to Long-Term Average (STA/LTA) algorithm.
- **Event identification**: Detects peaks in the seismic data, identifies significant events, and groups them for better clarity.
- **Seismic event characterization**: Further analyzes each detected event for additional properties like peak velocity and duration.
- **Dynamic thresholding**: Dynamically calculates thresholds based on data characteristics (mean and standard deviation).
- **Event visualization**: Plots both the velocity data and the STA/LTA ratio with detected events marked for easy visual inspection.
- **CSV export**: Exports the detected events, including metadata, into CSV files for future analysis.

## Data Sources

- **Apollo Seismic Data**: Contains continuous seismic records from the Apollo missions, which were active during lunar seismic experiments.
- **Mars InSight Lander Data**: Mars Interior Exploration using Seismic Investigations, Geodesy, and Heat Transport (InSight) Lander's seismic data, providing records from Mars’ surface.

## Approach

This project employs an STA/LTA algorithm to detect seismic events. It uses a sliding window approach to analyze short-term and long-term energy fluctuations in the data, allowing the detection of seismic events (quakes). The project is adaptable to missing data and glitches, which are common challenges in planetary data.

## Requirements

To run the project, the following libraries and dependencies are required:

### Python Packages:
- **Pandas**: For loading and managing seismic data from CSV files.
  - Install with: `pip install pandas`
- **NumPy**: For numerical operations on the seismic data arrays.
  - Install with: `pip install numpy`
- **Matplotlib**: For plotting seismic data and marking detected seismic events.
  - Install with: `pip install matplotlib`
- **OS Module**: Native Python module for handling file and directory operations (no need to install).

### Data Requirements:
- Seismic datasets in CSV format, with columns representing:
  - `time_rel(sec)`: Relative time in seconds.
  - `velocity(m/s)`: Seismic velocity data.
  - `time_abs(%Y-%m-%dT%H:%M:%S.%f)`: Absolute time formatted as a timestamp.

## Running the Program

### Step 1: Data Preparation
1. Place your seismic data CSV files in a folder named `datasets`. The program assumes this directory structure, but you can modify the directory name if needed.
2. Ensure each CSV file follows the standard format expected by the program (relative time, velocity, absolute time).

### Step 2: Install Dependencies
Install all the necessary Python packages:
```bash
pip install pandas numpy matplotlib
```

### Step 3: Run the Program
The program processes each file in the `datasets` folder using the following logic:
- Load seismic data from the CSV files.
- Apply the STA/LTA algorithm to identify potential seismic events.
- Dynamically compute a threshold for seismic event detection based on the characteristics of the data.
- Group detected events and characterize them (peak velocity, duration, total energy).
- Plot and save results for each file, showing both the seismic velocity data and the STA/LTA ratio, with detected events highlighted.
- Export detected seismic events into CSV files.

To execute the program, simply run:
```bash
python seismic_event_detection.py
```

### Output:
- **Plots**: The program generates visualizations of the seismic data, marking detected peaks and STA/LTA ratios.
- **CSV Files**: For each processed file, a corresponding CSV will be created with detected events, including:
  - Absolute Time of the seismic event.
  - Relative Time (in seconds).
  - Peak Velocity (in m/s).
  - Event Duration (in seconds).
  - Total Energy of the event.
  - The source file name.

### Example Output (Sample CSV):
```
Absolute Time,Relative Time (sec),Peak Velocity (m/s),Event Duration (sec),Total Energy,File
2024-10-05 12:34:56.789,123.45,0.067,10.2,0.12345,seismic_data_1.csv
2024-10-05 12:45:02.654,234.56,0.045,5.7,0.08934,seismic_data_1.csv
```

## Advanced Features
1. **Dynamic Threshold Calculation**: Instead of a static threshold, the program calculates a dynamic threshold using the mean and standard deviation of the STA/LTA ratio, allowing it to adapt to the signal's characteristics.
2. **Event Characterization**: Each seismic event is characterized by its peak velocity, duration, and energy, providing deeper insights into the detected seismic activities.
3. **Visualization**: The program provides clear and informative plots for each dataset, marking the identified seismic events with red "X" markers.

## Potential Improvements
- **Machine Learning Integration**: Machine learning techniques can be applied to improve the detection and classification of seismic events, especially when combined with training data.
- **Handling Glitches and Missing Data**: Additional algorithms can be developed to smooth or interpolate missing data, further enhancing event detection accuracy.

## License
This project is open-source. Feel free to modify and adapt it to suit your research needs.

---

## Contact

For further information or questions, feel free to contact:

- **Team Name**: [NOBODY]
- **Email**: [ariseonceagain@gmail.com]
- **GitHub**: [https://github.com/onceagainarise/Seismic_detection_across_solar_system]

---

## Conclusion

This project provides an efficient solution for analyzing planetary seismic data, helping identify seismic events while minimizing data transmission. By distinguishing signal from noise and only transmitting meaningful seismic events, this project can significantly reduce the power requirements for planetary seismology missions.

---

