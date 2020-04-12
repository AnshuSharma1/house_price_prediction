# House Price Prediction

The objective is to predict prices for a house given its features using linear regression and sequential learning

* Used Flask for prediction API
* Redis to store model data and scalers

## How to setup and run this api

* Create virtual environment in python3 and install libs from requirements.txt
    - `virtualenv -p python3 venv`
    - `sourve venv/bin/activate`
    - `pip install -r requirements.txt` 

* Run `python house_price_prediction.py` to fetch data from csv and store models in redis

* Run `price_prediction.sh` to run flask server where house prices can be predicted

* Once the server starts go to `http://127.0.0.1:5000/predict_prices/`
 - Predict house prices using LR and Sequential learning models for given set of features