import pandas as pd
from pandas.testing import assert_frame_equal
from datetime import datetime

from batch import prepare_data


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def test_prepare_data():
    expected_data = [
        {
            'PULocationID': '-1',
            'DOLocationID': '-1',
            'tpep_pickup_datetime': pd.Timestamp('2023-01-01 01:01:00'),
            'tpep_dropoff_datetime': pd.Timestamp('2023-01-01 01:10:00'),
            'duration': 9.0,

        },
        {
            'PULocationID': '1',
            'DOLocationID': '1',
            'tpep_pickup_datetime': pd.Timestamp('2023-01-01 01:02:00'),
            'tpep_dropoff_datetime': pd.Timestamp('2023-01-01 01:10:00'),
            'duration': 8.0,
        },
    ]
    expected_columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'duration']
    expected_df = pd.DataFrame(expected_data, columns=expected_columns)

    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    categorical = ['PULocationID', 'DOLocationID']

    df = pd.DataFrame(data, columns=columns)
    df = prepare_data(df, categorical)
    
    assert list(df.columns) == list(expected_df.columns)
    assert_frame_equal(df, expected_df)

