import os
import pickle

import numpy as np
import pandas as pd
import redis
from scipy import stats
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.layers import Dense

from contants import COLUMNS_TO_REMOVE, REDIS_CONNECTION_URL

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_file_data(file_name):
    filepath = os.path.join(BASE_DIR, file_name)
    csv_path = filepath + '.csv'
    data = pd.read_csv(csv_path, encoding='utf-8')
    return data


def get_redis_connection():
    return redis.Redis().from_url(REDIS_CONNECTION_URL)


def fill_null_data(feature_data):
    null_count = feature_data.isnull().sum()
    null_columns = null_count[null_count > 0].index
    if len(null_columns):
        feature_data.fillna(
            feature_data[null_columns].mean(),
            inplace=True
        )


def get_polynomial_features(features):
    # Expand feature set by polynomial combinations
    poly = PolynomialFeatures(2)
    poly_features = poly.fit_transform(features)
    poly_features = poly_features[:, 1:]
    poly_features = pd.DataFrame(poly_features)

    return poly_features


def get_tranformed_features(feature_data):
    # Get Relevant Columns
    feature_data['date'] = pd.to_datetime(feature_data['date'])
    feature_data['date'] = feature_data['date'].dt.year
    feature_data['orig_age'] = feature_data['date'] - feature_data['yr_built']
    feature_data.drop(COLUMNS_TO_REMOVE, axis=1, inplace=True)

    # Remove outliers on basis of z-score
    feature_data = feature_data[
        (np.abs(stats.zscore(feature_data)) < 3).all(axis=1)
    ]

    prices = feature_data['price'].values
    features = feature_data.drop(['price'], 1)

    # Normalise features between 0-1 scale
    scaler = MinMaxScaler()
    scaler.fit(features)
    feature_data = scaler.transform(features)

    return feature_data, prices, scaler


def get_linear_reg_model(
        feature_data,
        prices
):
    lm = LinearRegression()
    reg_model = lm.fit(feature_data, prices)

    return reg_model


def get_sequential_mode(
        feature_data,
        prices
):
    model = Sequential([
        Dense(64, activation='relu', input_shape=[10, ]),
        Dense(64, activation='relu'),
        Dense(1)
    ])

    optimizer = optimizers.RMSprop(0.001)
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])

    model.fit(
        feature_data,
        prices,
        epochs=50,
        shuffle=True
    )

    return model


def evaluate_model(model, test_features, prices):
    predictions = model.predict(test_features)
    mean_squared_error = np.sqrt(
        metrics.mean_squared_error(prices, predictions)
    )

    return mean_squared_error


def main():
    redis_con = get_redis_connection()
    house_data = get_file_data('kc_house_data')
    fill_null_data(house_data)
    feature_frame, price_frame, min_max_scaler = get_tranformed_features(house_data)
    train_features, test_features, train_price, test_price = train_test_split(
        feature_frame,
        price_frame,
        test_size=0.2
    )
    reg_model = get_linear_reg_model(train_features, train_price)
    seq_model = get_sequential_mode(train_features, train_price)
    reg_error = evaluate_model(reg_model, test_features, test_price)
    seq_error = evaluate_model(seq_model, test_features, test_price)

    print('Regression Model mean_squared_error', reg_error)
    print('Sequential Model mean_squared_error', seq_error)

    redis_con.set('reg_model', pickle.dumps(reg_model))
    redis_con.set('seq_model', pickle.dumps(seq_model.to_json()))
    redis_con.set('min_max_scaler', pickle.dumps(min_max_scaler))


if __name__ == "__main__":
    main()
