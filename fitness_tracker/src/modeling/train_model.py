import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix

# import LearningAlgorithms as LA
import sys

sys.path.append("/Users/abhijeetthombare/ab_lib/Projects/fitness_tracker/src/modeling/")
from LearningAlgorithms import ClassificationAlgorithms

# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


df = pd.read_pickle("../../data/interim/03_data_features.pkl")

# --------------------------------------------------------------
# Create a training and test set
# --------------------------------------------------------------

df_train = df.drop(["participant", "category", "set"], axis=1)

X = df_train.drop("label", axis=1)
y = df_train["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

fig, ax = plt.subplots(figsize=(10, 5))
df_train["label"].value_counts().plot(
    kind="bar", ax=ax, color="lightblue", label="Total"
)
y_train.value_counts().plot(kind="bar", ax=ax, color="dodgerblue", label="Train")
y_test.value_counts().plot(kind="bar", ax=ax, color="royalblue", label="Test")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Split feature subsets
# --------------------------------------------------------------

basic_features = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
square_features = ["acc_r", "gyr_r"]
pca_features = ["pca1", "pca2", "pca3"]
time_features = [f for f in df_train.columns if "_temp_" in f]
freq_features = [f for f in df_train.columns if (("_freq" in f) or ("_pse" in f))]
cluster_features = ["cluster"]

print("Basic Features: ", len(basic_features))
print("square Features: ", len(square_features))
print("PCA Features: ", len(pca_features))
print("Time Features: ", len(time_features))
print("Frequency Features: ", len(freq_features))
print("Cluster Features: ", len(cluster_features))

feature_set_1 = list(set(basic_features))
feature_set_2 = list(set(basic_features + square_features + pca_features))
feature_set_3 = list(set(feature_set_2 + time_features))
feature_set_4 = list(set(feature_set_3 + freq_features + cluster_features))

# --------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# --------------------------------------------------------------

# print(dir(LA))
max_features = 10
learner = ClassificationAlgorithms()

selected_features, ordered_features, ordered_scores = learner.forward_selection(
    max_features, X_train, y_train
)

"""Just in case if Kernel crashes we store values in veriables """
selected_features = [
    "pca_1",
    "duration",
    "acc_z_freq_0.0_Hz_ws_14",
    "gyr_r_freq_0.0_Hz_ws_14",
    "gyr_y_temp_std_ws_5",
    "gyr_y_freq_1.429_Hz_ws_14",
    "acc_z_freq_1.071_Hz_ws_14",
    "gyr_r",
    "acc_x_freq_2.5_Hz_ws_14",
    "acc_z_freq_0.714_Hz_ws_14",
]

ordered_features = [
    "pca_1",
    "duration",
    "acc_z_freq_0.0_Hz_ws_14",
    "gyr_r_freq_0.0_Hz_ws_14",
    "gyr_y_temp_std_ws_5",
    "gyr_y_freq_1.429_Hz_ws_14",
    "acc_z_freq_1.071_Hz_ws_14",
    "gyr_r",
    "acc_x_freq_2.5_Hz_ws_14",
    "acc_z_freq_0.714_Hz_ws_14",
]

ordered_scores = [
    0.8936651583710408,
    0.9773755656108597,
    0.997737556561086,
    0.9996767937944409,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
]

plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, max_features + 1, 1), ordered_scores)
plt.xlabel("Number of features")
plt.ylabel("Accuracy")
plt.xticks(np.arange(1, max_features + 1, 1))
plt.show()

# --------------------------------------------------------------
# Grid search for best hyperparameters and model selection
# --------------------------------------------------------------


# --------------------------------------------------------------
# Create a grouped bar plot to compare the results
# --------------------------------------------------------------


# --------------------------------------------------------------
# Select best model and evaluate results
# --------------------------------------------------------------


# --------------------------------------------------------------
# Select train and test data based on participant
# --------------------------------------------------------------


# --------------------------------------------------------------
# Use best model again and evaluate results
# --------------------------------------------------------------


# --------------------------------------------------------------
# Try a simpler model with the selected features
# --------------------------------------------------------------
