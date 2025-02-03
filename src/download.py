import os
import requests
import numpy as np
import pandas as pd
import h5py
import config as cfg

def download_ligo_data(url, data_path):
    """Download LIGO data if not already present.
    
    Parameters
    ----------
    url : str
        URL adress of the data set.
    data_path : str
        Path to the hdf5 file to be downloaded.
    """
    if not os.path.exists(data_path):
        print(f"Downloading data from {url}...")
        #downloading the file in parts to not load whole data into memory at once
        response = requests.get(url, stream=True)
        with open(data_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Download complete: {data_path}")
    else:
        print(f"File already exists: {data_path}")

def load_ligo_data(data_path, dataset_name="strain/Strain"):
    """Load LIGO strain data from an HDF5 file.
    
    Parameters
    ----------
    data_path : str
        Path to the hdf5 file.
    dataset_name : str
        Name of the dataset to be loaded.

    Returns
    -------
    return np.array
        Array that will be later transformed into a data frame.
    """
    with h5py.File(data_path, 'r') as f:
        strain = f[dataset_name][:]
    return np.array(strain)

def create_dataframe(strain, fs):
    """Create a Pandas DataFrame with time and strain.
    
    Parameters
    ----------
    strain : np.array
        Array of values corresponding to strain at times t.
    fs : int
        Sampling frequency of the LIGO detector.

    Returns
    -------
    return pd.DataFrame
        Data frame ready to be used in further analysis.
    """
    time = np.arange(0, len(strain) / fs, 1 / fs)
    df = pd.DataFrame({"time": time[:len(strain)], "strain": strain})
    return df

def save_dataframe(df, csv_path):
    """Save the DataFrame to a CSV file - done mostly for grading.

    Parameters
    ----------
    df : pd.DataFrame
        data frame to be saved - later uploaded to a VM.
    csv_path : str
        Path for the csv file to be saved.
    """
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
        print(f"Data saved to {csv_path}")
    else:
        print(f"File already exists: {csv_path}")

def get_data():
    download_ligo_data(cfg.URL, cfg.DATA_PATH)
    strain_data = load_ligo_data(cfg.DATA_PATH, "strain/Strain")
    ligo_df = create_dataframe(strain_data, cfg.FS)
    save_dataframe(ligo_df, cfg.CSV_PATH)
    return ligo_df

if __name__ == "__main__":
    get_data()