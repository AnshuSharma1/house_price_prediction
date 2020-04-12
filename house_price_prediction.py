import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

from contants import COLUMNS_TO_REMOVE
from sklearn.linear_model import LinearRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_file_data(file_name):
    filepath = os.path.join(BASE_DIR, file_name)
    csv_path = filepath + '.csv'
    data = pd.read_csv(csv_path, encoding='utf-8')
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
    # Get Relevant Columns
    feature_data['date'] = pd.to_datetime(feature_data['date']).dt.year
    feature_data['orig_age'] = feature_data['date'] - feature_data['yr_built']
    feature_data.drop(COLUMNS_TO_REMOVE, axis=1, inplace=True)

    # Remove outliers on basis of z-score
    feature_data = feature_data[
        (np.abs(stats.zscore(feature_data)) < 3).all(axis=1)
    ]

    # Normalise features between 0-1 scale
    scaler = MinMaxScaler()
    feature_data = scaler.fit_transform(feature_data)

    prices = feature_data['price']
    features = feature_data[:, 1:]

    # Expand feature set by polynomial combinations
    poly = PolynomialFeatures(2)
    poly_features = poly.fit_transform(features)
    poly_features = poly_features[:, 1:]
    poly_features = pd.DataFrame(poly_features)

    return poly_features, prices


def get_linear_reg_model(
        feature_data,
        prices
):
    lm = LinearRegression()
    reg_model = lm.fit(feature_data, prices)

    return reg_model


def main():
    house_data = get_file_data('kc_house_data')
    fill_null_data(house_data)
    feature_frame,  price_frame = get_tranformed_features(house_data)
    reg_model = get_linear_reg_model(feature_frame, price_frame)


if __name__ == "__main__":
    main()
