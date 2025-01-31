import matplotlib.pyplot as plt
import pandas as pd
import config as cfg

def plot_dataframe(df):
    """
    Plot original and processed signal. Resulting png file is saved.
    Parameters
    ----------
    df : pd.DataFrame
        Data to be plotted.
    """
    plt.figure()

    plt.subplot(3, 1, 1)
    plt.plot(df["time"], df["strain"], color='gray', label="Original Signal")
    plt.ylabel("Strain")
    plt.xlabel("Time [s]")
    plt.legend()

    plt.show()

    plt.savefig(cfg.PLOTS_PATH)