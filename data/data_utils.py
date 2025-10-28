import os
import pandas as pd
import matplotlib.pyplot as plt

PATH = os.path.join(os.path.dirname(__file__), "cancer-risk-factors.csv")

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


