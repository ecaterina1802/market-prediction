# create_csv.py

import pandas as pd

# Updated sample data for the CSV file
data = {
    "Date": ["2024-10-01", "2024-10-02", "2024-10-03", "2024-10-04", "2024-10-05",
             "2024-10-06", "2024-10-07", "2024-10-08", "2024-10-09", "2024-10-10"],
    "Open": [130.50, 131.50, 132.00, 132.80, 133.00, 133.80, 134.10, 134.50, 135.10, 135.50],
    "High": [132.00, 132.30, 133.00, 133.60, 134.20, 134.50, 135.00, 135.50, 136.00, 136.50],
    "Low": [129.50, 130.00, 131.10, 131.40, 132.50, 133.00, 133.20, 133.50, 134.20, 134.70],
    "Close": [131.20, 131.80, 132.50, 132.90, 133.50, 134.00, 134.50, 135.00, 135.80, 136.20],
    "Volume": [700000, 750000, 800000, 770000, 740000, 710000, 720000, 730000, 740000, 750000]
}

# Create a DataFrame using the updated data
df = pd.DataFrame(data)

# Write the DataFrame to a CSV file
df.to_csv('your_dataset.csv', index=False)

print("CSV file 'your_dataset.csv' has been created with updated data.")
