#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import sys

year = int(sys.argv[1]) # 2023
month = int(sys.argv[2]) # 4
model_path = sys.argv[3]


with open(model_path, 'rb') as f_in:
    dv, model = pickle.load(f_in)


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet')


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)
y_pred.std()
print(y_pred.mean())

df['ride_id'] = f'2023/03_' + df.index.astype('str')

df_q2 = pd.DataFrame()
df_q2['ride_id'] = df['ride_id']
df_q2['predicted_duration'] = y_pred
df_q2.to_parquet(
    'q2_output_file',
    engine='pyarrow',
    compression=None,
    index=False
)
