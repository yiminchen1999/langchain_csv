import pandas as pd
import os

# Specify the folder path containing the CSV files
folder_path = "merge"

# Get a list of all CSV files in the folder
csv_files = [file for file in os.listdir(folder_path) if file.endswith(".csv")]

# Initialize an empty DataFrame to store the merged data
merged_df = pd.DataFrame()

# Loop through each CSV file and merge its data into the merged DataFrame
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    merged_df = pd.concat([merged_df, df], ignore_index=True)

# Save the merged DataFrame to a new CSV file
merged_df.to_csv("merged_dataset.csv", index=False)

print("Merged dataset saved as merged_dataset.csv")

