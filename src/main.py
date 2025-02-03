from download import get_data
from signal_processing import process_dataframe
from plot import plot_all_strains

def main():
    """
    Main function to process and analyze LIGO strain data.
    """
    df = get_data()
    df = process_dataframe(df)
    plot_all_strains(df)

if __name__ == "__main__":
    main()