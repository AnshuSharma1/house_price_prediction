import pickle

import redis
from flask import Flask, request, jsonify
from tensorflow.keras.models import model_from_json

from contants import REDIS_CONNECTION_URL

app = Flask(__name__)


def get_redis_connection():
    return redis.Redis().from_url(REDIS_CONNECTION_URL)


redis_con = get_redis_connection()


@app.route('/')
def root():
    url_list = [
        request.url + 'predict_prices/?bedrooms=3&bathrooms=1&sqft_living=1180&sqft_lot=5650&'
                      'floors=1&condition=3&grade=7&sqft_above=1180&sqft_basement=0&orig_age=59'
    ]
    return jsonify(url_list)


@app.route('/predict_prices/')
def predict():
    """
    Fetches model data from redis and predict house prices
    :return: JSON response of predicted data
    """
    bedrooms = int(request.args.get('bedrooms', None))
    sqft_living = int(request.args.get('sqft_living', None))
    floors = int(request.args.get('floors', None))
    orig_age = int(request.args.get('orig_age', None))
    bathrooms = int(request.args.get('bathrooms', 1))
    sqft_lot = int(request.args.get('sqft_lot', 5650))
    condition = int(request.args.get('condition', 3))
    grade = int(request.args.get('grade', 7))
    sqft_above = int(request.args.get('sqft_above', 1180))
    sqft_basement = int(request.args.get('sqft_basement', 0))

    if None in [bedrooms, sqft_living, floors, orig_age]:
        return 'Insufficient args'

    try:
        reg_model = pickle.loads(redis_con.get('reg_model'))
        seq_model = pickle.loads(redis_con.get('seq_model'))
        seq_model = model_from_json(seq_model)
        min_max_scaler = pickle.loads(redis_con.get('min_max_scaler'))
    except TypeError:
        return 'Key does not exist'

    # Scale given data using same scaler used in training
    scaled_data = min_max_scaler.transform([[bedrooms, bathrooms, sqft_living,
                                             sqft_lot, floors, condition, grade,
                                             sqft_above, sqft_basement, orig_age]])

    reg_price = float(reg_model.predict(scaled_data)[0])
    seq_price = float(seq_model.predict(scaled_data)[0][0])

    response = {
        'reg_predicted_price': reg_price,
        'seq_predicted_price': seq_price
    }

    return jsonify(response)


if __name__ == '__main__':
    # Hit http://127.0.0.1:5000/?
    app.run(debug=True)
