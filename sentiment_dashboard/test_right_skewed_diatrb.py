import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # for KDE plot

# Right-skewed data due to the outlier 100
data = pd.Series([1, 2, 2, 3, 3, 3, 4, 5, 100])
print("Skewness:", data.skew())


""" Density Plot / KDE"""

# KDE plot
sns.kdeplot(data, fill=True, color="skyblue", bw_adjust=1)
plt.title("KDE Plot (Smoothed Curve) of Right-Skewed Data")
plt.xlabel("Value")
plt.ylabel("Density")
plt.xticks(np.arange(0, 105, 5))
plt.grid(True)
plt.tight_layout()
plt.show()

# Plotting histogram
plt.hist(data, bins=10, edgecolor="black")
plt.title("Histogram of Right-Skewed Data")
plt.xlabel("Value")
plt.ylabel("Frequency")
# Custom x-axis ticks: from 0 to 105 with a step of 5
plt.xticks(np.arange(0, 105, 5))

plt.grid(True)
plt.tight_layout()
plt.show()

"""line plot"""
sorted_data = data.sort_values().reset_index(drop=True)

plt.plot(sorted_data, marker="o", linestyle="-")
plt.title("Line Plot of Sorted Values (Right-Skewed Data)")
plt.xlabel("Index (sorted)")
plt.ylabel("Value")

# Optional: Add horizontal grid and x-ticks for readability
plt.yticks(np.arange(0, 105, 5))
plt.grid(True)
plt.tight_layout()
plt.show()


"""   option 2 line plot  """

cumulative = sorted_data.cumsum()

plt.plot(cumulative, marker="o", linestyle="--", color="purple")
plt.title("Cumulative Sum Line Plot")
plt.xlabel("Index (sorted)")
plt.ylabel("Cumulative Value")
plt.grid(True)
plt.tight_layout()
plt.show()


""" scattered plot """

# X-axis: index; Y-axis: value
plt.scatter(data.index, data.values, color="blue", marker="o")
plt.title("Scatter Plot of Right-Skewed Data")
plt.xlabel("Index")
plt.ylabel("Value")
plt.yticks(np.arange(0, 105, 5))  # Y-axis ticks from 0 to 100
plt.grid(True)
plt.tight_layout()
plt.show()
