import os
import pandas as pd

PATH = os.path.join(os.path.dirname(__file__), "Cancer_Data.csv")

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(PATH)
    data_cleaned = data.dropna(axis=1, how="all")
    return data, data_cleaned


