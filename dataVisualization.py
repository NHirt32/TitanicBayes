import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.figure(figsize=(12,10))

def heatmap(dataFrame):
    corrMatrix = dataFrame.corr()
    sns.heatmap(corrMatrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.show()
