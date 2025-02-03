import matplotlib.pyplot as plt
import pandas as pd
import config as cfg

def plot_all_strains(df):
    """
    Plot original and processed strain signals in time.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame containing columns ["time", "strain", "processed", "processed_differently"].

    """
    df_segment = df[(df["time"] >= 0.3) & (df["time"] <= 0.45)]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    colors = ["#D8BFD8", "#FFB6C1", "#ADD8E6"]  # Light Purple, Baby Pink, Baby Blue
    labels = ["Original Strain", "Processed (Custom)", "Processed (GWpy)"]
    columns = ["strain", "processed", "processed_differently"]

    for ax, col, color, label in zip(axes, columns, colors, labels):
        ax.plot(df_segment["time"], df_segment[col], color=color, label=label, alpha=0.9)
        ax.set_ylabel("Strain")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("LIGO Strain Signals Over Time (0.3s - 0.45s)", fontsize=14)
    plt.tight_layout()

    plt.savefig(cfg.PLOTS_PATH, dpi=300, bbox_inches="tight")
    print(f"Plot saved as {cfg.PLOTS_PATH}")
