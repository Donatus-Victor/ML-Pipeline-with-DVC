import os
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


# Function to get the raw data
def load_data(data_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)  # Read the CSV data from a link or folder
        return df  
    except pd.errors.ParserError as e:  # Handle errors if the file format is broken
        print(f"Error: Failed to parse the CSV file from {data_url}.") 
        print(e)  # Print the exact format error
        raise  # Stop the code so the broken pipeline doesn't continue
    except Exception as e:  # Handle any other sudden problems (like network or missing file)
        print(f"Error: An unexpected error occurred while loading the data.")  
        print(e)  # Print the exact system error
        raise  # Stop the code immediately to prevent blank data bugs
    

# Defining a step to transform and clean data into an ML-ready state
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and filter data into binary classes for ML training."""
    try:
        # Drop identifier column that lacks predictive value
        df = df.drop(columns=["tweet_id"])
        
        # Explicitly copy filtered data to avoid SettingWithCopyWarning
        final_df = df[df["sentiment"].isin(["happiness", "sadness"])].copy()
        
        # Convert target categories to 1s and 0s via direct assignment
        final_df["sentiment"] = final_df["sentiment"].replace({"happiness": 1, "sadness": 0})
        
        # Avoid future deprecation warnings by handling data downcasting safely
        final_df = final_df.infer_objects(copy=False)

        return final_df
    except KeyError as e:  # Guard against missing expected columns if schema changes
        print(f"Error: Missing expected column {e} in the dataframe.")
        raise   # raise will crash the  pipeline to prevent downstream training on corrupted schemas
    except Exception as e:  # to catch unforeseen issues (e.g., trying to modify a view instead of a copy)
        print(f"Error: An unexpected error occurred during preprocessing.\n{e}")
        raise ## raise will break the process execution chain


# Save the train and test data to a folder

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save train and test splits into a local folder destination."""
    try:
        target_dir = os.path.join(data_path, "raw")
        os.makedirs(target_dir, exist_ok=True)
        
        # Save files as train and test split without dumping the pandas index column
        train_data.to_csv(os.path.join(target_dir, "train.csv"), index=False)
        test_data.to_csv(os.path.join(target_dir, "test.csv"), index=False)
        
    except Exception as e:
        print(f"Error: An unexpected error occurred while saving the data.\n{e}")
        raise # Crash to prevent downstream tasks from reading incomplete or missing datasets



def main():
    try:
        # Fetch the remote source data path
        url = "https://raw.githubusercontent.com/Donatus-Victor/ML-Pipeline-with-DVC/refs/heads/main/tweet_emotions.csv"
        df = load_data(url)

        # Process, split, and save data splits
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=0.2, random_state=42)
        save_data(train_data, test_data, data_path="data")
        
        print("Data ingestion and processing completed successfully.")
    except Exception as e:
        print(f"Error: {e}")
        print("Failed to complete the data ingestion process.")
        
        
# Prevent code from running automatically if this file is imported as a modular helper tool elsewhere
if __name__ == "__main__":
    main()
