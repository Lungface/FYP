import os
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np

# Define the base directory containing the sessions
base_directory = 'Test_converted'
output_directory = 'Gesture_Plots_Output'

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Loop through all sessions (Session1, Session2, Session3)
for session in range(1, 4):  # Sessions 1 to 3
    input_directory = os.path.join(base_directory, f'Session{session}_converted')
    
    if not os.path.exists(input_directory):
        print(f"Session directory {input_directory} does not exist. Skipping...")
        continue

    # Loop through all .mat files in the session directory
    for mat_file in os.listdir(input_directory):
        if mat_file.endswith('.mat'):  # Check if the file is a .mat file
            mat_file_path = os.path.join(input_directory, mat_file)
            
            # Load the .mat file
            mat_data = loadmat(mat_file_path)
            
            # Process both DATA_FOREARM and DATA_WRIST
            for data_key in ['DATA_FOREARM', 'DATA_WRIST']:
                if data_key not in mat_data:
                    print(f"{data_key} not found in {mat_file}. Skipping...")
                    continue
                
                data = mat_data[data_key]
                print(f"Processing file: {mat_file}, Dataset: {data_key}")
                print("Num Trials:", data.shape[0], "Num Gestures:", data.shape[1])
                
                # Loop through all trials and gestures
                for trial in range(data.shape[0]):  # Loop through trials
                    for gesture in range(data.shape[1]):  # Loop through gestures
                        data_to_plot = data[trial, gesture]
                        print(f"Trial {trial + 1}, Gesture {gesture + 1}")
                        print("Timesteps:", data_to_plot.shape[0], "Num Channels:", data_to_plot.shape[1])
                        
                        # Loop through all channels
                        for ichannel in range(data_to_plot.shape[1]):  # Loop through channels
                            time = range(data_to_plot.shape[0])
                            
                            # Create a folder for the current gesture if it doesn't exist
                            gesture_folder = os.path.join(output_directory, f"Session_{session}", data_key, f"Gesture_{gesture + 1}")
                            if not os.path.exists(gesture_folder):
                                os.makedirs(gesture_folder)
                            
                            # Save the plot
                            plt.figure(figsize=(10, 6))
                            plt.plot(time, data_to_plot[:, ichannel])
                            plt.title(f'File: {mat_file}, Dataset: {data_key}, Trial {trial + 1}, Gesture {gesture + 1}, Channel {ichannel + 1}')
                            plt.xlabel('Time')
                            plt.ylabel('Amplitude')
                            plt.grid(True)
                            
                            # Save the plot as a PNG file
                            plot_filename = f"{os.path.splitext(mat_file)[0]}_Trial{trial + 1}_Gesture{gesture + 1}_Channel{ichannel + 1}.png"
                            plot_filepath = os.path.join(gesture_folder, plot_filename)
                            plt.savefig(plot_filepath)
                            plt.close()  # Close the plot to free memory