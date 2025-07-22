import numpy as np
import pandas as pd
import os
import yaml

from sklearn.model_selection import train_test_split

# Load parameters from params.yaml
with open('params.yaml', 'r') as file:
    params = yaml.safe_load(file)

test_size = params['data_ingestion']['test_size']

# Read the dataset directly from the provided URL
df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')

# Drop the 'tweet_id' column as it is not needed for modeling
df.drop(columns=['tweet_id'], inplace=True)

# Filter the DataFrame to keep only rows with 'happiness' or 'sadness' sentiments
final_df = df[df['sentiment'].isin(['happiness', 'sadness'])]

# Convert sentiment labels to binary: 'happiness' -> 1, 'sadness' -> 0
final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)

# Split the data into training and testing sets (80% train, 20% test)
train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

# Create the directory to save raw data if it doesn't exist
os.makedirs('data/raw', exist_ok=True)

# Save the train and test splits as CSV files
train_data.to_csv('data/raw/train.csv', index=False)
test_data.to_csv('data/raw/test.csv', index=False)
