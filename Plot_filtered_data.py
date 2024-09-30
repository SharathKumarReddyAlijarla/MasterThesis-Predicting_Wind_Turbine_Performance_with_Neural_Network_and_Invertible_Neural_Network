import os
import matplotlib.pyplot as plt


# Function to plot wind speed vs. pitch angle from CSV filenames in the given directory.
# This function extracts wind speed and pitch angle values from the filenames and plots them.
def plot_wind_speed_pitch_angle(output_directory):
    # Lists to store wind speeds and corresponding pitch angles extracted from filenames.
    wind_speeds = []
    pitch_angles = []

    # Loop through each file in the specified directory.
    for filename in os.listdir(output_directory):
        # Check if the file is a CSV file.
        if filename.endswith(".csv"):
            # Split the filename into parts to extract wind speed and pitch angle information.
            filename_parts = filename.split("_")
            wind_speed = None  # Variable to store the wind speed value.
            pitch_angle = None  # Variable to store the pitch angle value.

            # Loop through parts of the filename to find wind speed and pitch angle.
            for i, part in enumerate(filename_parts):
                # Extract wind speed if the filename part starts with 'wind'.
                if part.startswith("wind") and i < len(filename_parts) - 1:
                    try:
                        # Convert the next part after 'wind' to a float.
                        wind_speed = float(filename_parts[i + 1])
                    except ValueError:
                        print("Invalid wind speed format:", filename_parts[i + 1])

                # Extract pitch angle if the filename part starts with 'p'.
                elif part.startswith("p"):
                    try:
                        # Convert the substring after 'p' to a float (e.g., 'p10' -> 10).
                        pitch_angle = float(part[1:])
                    except ValueError:
                        print("Invalid pitch angle format:", part)

            # If both wind speed and pitch angle were successfully extracted, store them in the lists.
            if wind_speed is not None and pitch_angle is not None:
                wind_speeds.append(wind_speed)
                pitch_angles.append(pitch_angle)

    # Create a scatter plot of wind speed vs. pitch angle.
    plt.figure(figsize=(10, 6))  # Set the figure size.
    plt.plot(wind_speeds, pitch_angles, 'bo', markersize=3)  # Plot the data points with blue circles.
    plt.xlabel('Wind Speed (m/s)')  # Set the x-axis label.
    plt.ylabel('Pitch Angle')  # Set the y-axis label.
    plt.title('Wind Speed vs Pitch Angle')  # Set the plot title.
    plt.grid(True)  # Enable the grid for better visualization.
    plt.show()  # Display the plot.


# Example usage: Replace the path with the directory containing your CSV files.
output_directory = r'C:\Users\alijarla\Desktop\Operational_data'
plot_wind_speed_pitch_angle(output_directory)
