import os
import pandas as pd
import re
import shutil


# Function to filter and copy CSV files based on specified conditions.
# It reads an Excel file with ideal values and compares the filenames of CSV files
# against the ideal values for wind speed, pitch angle, and generator speed.
def filter_csv_files(csv_directory, ideal_values_file, output_directory):
    """
    Filters and copies CSV files from the source directory to the destination directory
    based on conditions related to wind speed, pitch angle, and generator speed.

    Args:
        csv_directory (str): Path to the directory containing the CSV files.
        ideal_values_file (str): Path to the Excel file containing ideal wind speed, pitch angle, and speed values.
        output_directory (str): Path to the directory where filtered CSV files will be copied.

    Returns:
        None
    """
    # Load ideal values from the provided Excel file into a DataFrame.
    ideal_values_df = pd.read_excel(ideal_values_file, engine='openpyxl')

    # Create the output directory if it doesn't exist.
    os.makedirs(output_directory, exist_ok=True)
    
    # Iterate through each file in the source directory.
    for filename in os.listdir(csv_directory):
        # Process only CSV files.
        if filename.endswith(".csv"):
            file_path = os.path.join(csv_directory, filename)
            filename_parts = filename.split("_")  # Split filename into parts based on underscores.

            # Ensure the filename has enough parts to extract wind speed, pitch angle, and speed.
            if len(filename_parts) >= 7:
                # Initialize indices for identifying positions of parameters in the filename.
                wind_speed_index, pitch_angle_index, gen_speed_index = -1, -1, -1

                # Loop through the parts of the filename to find indices of wind speed, pitch angle, and speed.
                for i, part in enumerate(filename_parts):
                    # Identify the index of wind speed in the filename.
                    if 'wind' in part:
                        wind_speed_index = i
                    # Identify the index of pitch angle if it starts with 'p' and is a valid number.
                    elif 'p' in part and part[1:].replace('.', '').isdigit():
                        pitch_angle_index = i
                    # Identify the index of generator speed if it matches the 'sp' pattern.
                    elif re.match(r'sp\d+(\.\d+)?\.csv', part):
                        gen_speed_index = i
                        
                # Check if all necessary indices were found.
                if wind_speed_index != -1 and pitch_angle_index != -1 and gen_speed_index != -1:
                    # Extract wind speed, pitch angle, and generator speed from the filename.
                    wind_speed = filename_parts[wind_speed_index + 1]
                    pitch_angle = filename_parts[pitch_angle_index][1:]  # Remove 'p' prefix from pitch angle.
                    Sp = filename_parts[gen_speed_index].replace('sp', '').replace('.csv', '')

                    # Retrieve the row from the ideal values DataFrame matching the current wind speed.
                    wind_speed_row = ideal_values_df[ideal_values_df['wind.param.vHub_[m/s]'] == float(wind_speed)]

                    # Check if a matching row for the wind speed exists in the ideal values.
                    if not wind_speed_row.empty:
                        true_pitch_angle = wind_speed_row['pitch angle'].values[0]  # Ideal pitch angle.
                        true_SP = wind_speed_row['sp'].values[0]  # Ideal speed (sp).

                        # Check if the pitch angle and speed in the filename are within 30% tolerance of the ideal values.
                        if abs(float(pitch_angle) - abs(true_pitch_angle)) <= 0.3 * abs(true_pitch_angle):
                            if abs(float(Sp) - true_SP) <= 0.3 * true_SP:
                                # Copy the file to the output directory if it meets all conditions.
                                destination_path = os.path.join(output_directory, filename)
                                shutil.copy(file_path, destination_path)
                else:
                    # If required parameters are missing from the filename, skip the file.
                    print("Skipping file due to missing parameters:", filename)
            else:
                # Skip the file if it doesn't have the required number of parts.
                print("Skipping file due to insufficient parts in filename:", filename)


# Example usage: Define directory paths for CSV files, ideal values file, and output directory.
csv_directory = r'E:\CSVFiles'
ideal_values_file = r'C:\Users\alijarla\Desktop\results\pitchangles.xlsx'
output_directory = r'C:\Users\alijarla\Desktop\op_data_30tol'

# Call the function to filter CSV files based on the conditions.
filter_csv_files(csv_directory, ideal_values_file, output_directory)
