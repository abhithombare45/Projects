import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def load_data():
    """Load and preprocess the posts data"""
    df = pd.read_csv("./posts.csv")
    # Convert Time column to datetime
    df["Time"] = pd.to_datetime(df["Time"])
    return df


def visualize_posting_frequency(df, freq="D"):
    """
    Visualize posting frequency over time
    freq: resampling frequency ('H'=hour, 'D'=day, 'W'=week)
    """
    # Resample by frequency and count posts
    freq_df = df.resample(freq, on="Time").count()

    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(freq_df.index, freq_df["Title"], marker="o")
    plt.title(f"Posting Frequency (per {freq})")
    plt.xlabel("Date")
    plt.ylabel("Number of Posts")
    plt.grid(True)
    plt.tight_layout()
    # Save plot to file
    plot_path = "./posting_frequency.png"
    plt.savefig(plot_path)
    print(f"Visualization saved to {plot_path}")
    print("Sample data:")
    print(freq_df.head())


if __name__ == "__main__":
    df = load_data()
    visualize_posting_frequency(df, freq="D")
