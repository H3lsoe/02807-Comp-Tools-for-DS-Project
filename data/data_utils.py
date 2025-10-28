# -------------------------------Imports---------------------------------- # 

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# -------------------------------Constants-------------------------------- # 

PATH = os.path.join(os.path.dirname(__file__), "cancer-risk-factors.csv")
TARGET_COL = "Cancer_Type"
DROP_COLS = ["Patient_ID", "Cancer_Type", "Overall_Risk_Score", "Risk_Level"]

# -------------------------------Functions-------------------------------- #

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(PATH)
    data_cleaned = data.dropna(axis=1, how="all")
    return data, data_cleaned

def display_data(data, mapping=[]):
    xs, ys = zip(*data)
    if len(mapping) != 0:
        plt.scatter(xs,ys, c=mapping, cmap='tab10')
        plt.show()
    else:
        plt.scatter(xs,ys)
        plt.show()


def train_val_test_split(
    data: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
):
    # Separate features/target
    y = data[TARGET_COL]
    X = data.drop(columns=DROP_COLS)

    # One-hot encode categorical features automatically
    X = pd.get_dummies(X, drop_first=True) # yes/no -> 1/0 - ved ikke om der er nogen, men for en sikkerheds skyld
    
    # Split into train/val/test
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_size, stratify=y_tmp, random_state=random_state
    )

    return X_train, y_train, X_val, y_val, X_test, y_test