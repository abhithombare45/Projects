import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------

set_df = df[df["set"] == 1]

plt.plot(set_df["acc_x"])  # now compare with the df   # plt.plot(df["acc_x"])

plt.plot(set_df["acc_x"].reset_index(drop=True)) 

# Aswe are not understanding what it showes We plot as per label
# --------------------------------------------------------------
# Plot all exercises/labes
# --------------------------------------------------------------

for label in df["label"].unique():
    subset = df[df["label"] == label]
    # display(subset.head(2))
    plt.plot(subset["acc_x"].reset_index(drop=True), label=label) # we working with all datapoints on acc_x
    plt.legend()
    plt.show()

for label in df["label"].unique():
    subset = df[df["label"] == label]
    # display(subset.head(2))
    plt.plot(subset[:100]["acc_x"].reset_index(drop=True), label=label) # we added first 100 itteration of acc_x
    plt.legend()
    plt.show()

# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------

mpl.style.use("seaborn-v0-_8-deep")
mpl.rcParams("figure.figsize") = [20,5]
# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------


# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------


# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------


# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------


# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------


# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------
