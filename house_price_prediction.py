import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_file_data(file_name):
    filepath = os.path.join(BASE_DIR, file_name)
    csv_path = filepath + '.csv'
    data = pd.read_csv(csv_path, encoding='utf-8')
    data.iloc[0, 3] = None
    return data


def fill_null_data(feature_data):
    null_count = feature_data.isnull().sum()
    null_columns = null_count[null_count > 0].index
    if len(null_columns):
        feature_data.fillna(
            feature_data[null_columns].mean(),
            inplace=True
        )


def get_tranformed_features(feature_data):
    feature_data['date'] = pd.to_datetime(feature_data['date']).year
    feature_data['orig_age'] = feature_data['date'] - feature_data['yr_built']
    columns_to_remove = ['id', 'date', 'yr_built', 'yr_renovated']
    feature_data.drop(columns_to_remove, axis=1, inplace=True)

    feature_data = feature_data[(np.abs(stats.zscore(feature_data)) < 3).all(axis=1)]
    columns_to_normalise = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                            'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement',
                            'orig_age', 'sqft_living15', 'sqft_lot15'
                            ]
    scaler = MinMaxScaler()
    feature_data.loc[:, columns_to_normalise] = scaler.fit_transform(feature_data.loc[:, columns_to_normalise])

    return feature_data


def clean_feature_data(frame):
    fill_null_data(frame)
    frame = get_tranformed_features(frame)


def main():
    house_data = get_file_data('kc_house_data')
    clean_feature_data(house_data)


if __name__ == "__main__":
    main()
