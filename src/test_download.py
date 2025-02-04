import os
import config as cfg
import download as dwnld
import numpy as np
import pandas as pd
import pytest

def test_download_ligo_data_file_exists():
    """
    Test the download_ligo_data function when a file already exists at the specified path.
    """
    temp_data_path = "/tmp/test_data.hdf5"
    try:
        dwnld.download_ligo_data(cfg.URL, temp_data_path)
    except AssertionError as e:
        assert str(e) == "File already exists: {temp_data_path}"
    finally:
        os.remove(temp_data_path)

def test_load_ligo_data_error():
    """
    Test the load_ligo_data function when the specified file does not exist.
    """
    non_existent_file_path = "/tmp/non_existent_data.hdf5"
    try:
        dwnld.load_ligo_data(non_existent_file_path, "strain/Strain")
    except RuntimeError as e:
        assert str(e) == "File not found at /tmp/non_existent_data.hdf5"

def test_load_ligo_data_dataset_error():
    """
    Test the load_ligo_data function when the specified dataset does not exist in the file.
    """
    temp_data_path = "/tmp/temp.hdf5"
    try:
        dwnld.load_ligo_data(temp_data_path, "strain/NonExistentDataset")
    except RuntimeError as e:
        assert str(e) == "Dataset 'strain/NonExistentDataset' not found in file"

def test_create_dataframe_error_empty_strain():
    """
    Test the create_dataframe function when the provided strain data is empty.
    """
    try:
        dwnld.create_dataframe([], cfg.FS)
    except TypeError as e:
        assert str(e) == "Strain data must be a np.array."

def test_create_dataframe_error_non_positive_fs():
    try:
        dwnld.create_dataframe(np.array([0.1, 0.2, 0.3]), -1)
    except ValueError as e:
        assert str(e) == "Sampling frequency must be a positive integer."

def test_create_dataframe_error_mismatched_lengths():
    """
    Test the create_dataframe function when the provided time and strain lengths do not match.
    """
    try:
        dwnld.create_dataframe(np.array([0.1, 0.2, 0.3]), 10)
    except ValueError as e:
        assert str(e) == "The length of the time array does not match the length of the strain array"

def test_save_dataframe_file_exists(tmpdir):
    """
    Test the save_dataframe function when a file already exists at the specified path.
    """
    temp_csv_path = str(tmpdir.join("test_data.csv"))
    try:
        dwnld.save_dataframe(pd.DataFrame(), temp_csv_path)
    except AssertionError as e:
        assert str(e) == "File already exists: {temp_csv_path}"

def test_save_dataframe_invalid_extension(tmpdir):
    """
    Test the save_dataframe function when an invalid file extension is provided.
    """
    invalid_csv_path = str(tmpdir.join("test_data.txt"))
    try:
        dwnld.save_dataframe(pd.DataFrame(), invalid_csv_path)
    except ValueError as e:
        assert str(e) == "The provided CSV path does not have a .csv extension"

if __name__ == "__main__":
    pytest.main(['tests/test_download.py'])
