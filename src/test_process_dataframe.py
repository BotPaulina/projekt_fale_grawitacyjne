import pandas as pd
import numpy as np
import config as cfg
import pytest
from signal_processing import process_dataframe


def test_process_dataframe_working_condition():
    """
    Test the process_dataframe function under normal conditions.
    """
    df = pd.DataFrame(pd.read_csv(cfg.CSV_PATH))
    processed_df = process_dataframe(df)

    assert 'processed' in processed_df.columns
    assert 'processed_differently' in processed_df.columns

def test_process_dataframe_missing_columns():
    """
    Test the process_dataframe function when required columns are missing.
    """
    data = {
        'time': np.arange(1000)
    }
    df = pd.DataFrame(data)

    with pytest.raises(KeyError):
        process_dataframe(df)

def test_process_dataframe_invalid_data_type():
    """
    Test the process_dataframe function when input data is not a pandas DataFrame.
    """
    invalid_data_types = [np.array([1, 2, 3]), pd.Series([1, 2, 3])]

    for data in invalid_data_types:
        with pytest.raises(TypeError):
            process_dataframe(data)

if __name__ == "__main__":
    pytest.main()