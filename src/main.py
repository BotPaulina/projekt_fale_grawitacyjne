from download import main as download_main
from signal_processing import process_dataframe
from plot import plot_dataframe
import config as cfg

def main():
    """
    Main function to process and analyze LIGO strain data.
    """
    df = download_main()
    df = process_dataframe(df)
    plot_dataframe(df)

if __name__ == "__main__":
    main()