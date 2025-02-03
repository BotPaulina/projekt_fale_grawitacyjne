from download import get_data
from signal_processing import process_dataframe
from plot import plot_all_strains

def main():
    """
    Main function to process and analyze LIGO strain data.
    """
    try:
        df = get_data()
        if df is None:
            print("Failed to download or load data.")
            return

        df_processed = process_dataframe(df)

        if df_processed is None:
            print("Failed to process data.")
            return

        plot_all_strains(df_processed)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()