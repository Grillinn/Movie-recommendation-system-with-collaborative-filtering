import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Load movie ratings data from a CSV file.
    """
    return pd.read_csv(file_path)

def preprocess_data(data):
    """
    Preprocess the data by removing duplicates and handling missing values.
    """
    data.drop_duplicates(inplace=True)
    data.dropna(inplace=True)
    return data

def split_data(data, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.
    """
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    return train_data, test_data