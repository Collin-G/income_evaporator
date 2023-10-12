import os
import statistics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from pandas_datareader import data as pdr
import datetime as dt
from datetime import datetime, timedelta

from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import yfinance as yfin

def x_and_y_train(predict_length,data):
    x_train = []
    y_train = []

    for i in range(60, len(data)):
        x_train.append(data[i-predict_length:i,0])
        y_train.append(data[i,0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
    return x_train, y_train

def build_model(predict_length, data):

    x_train, y_train = x_and_y_train(predict_length, data)
    model = Sequential()
    model.add(LSTM(units=50, return_sequences = True, input_shape=(x_train.shape[1],1)))
    model.add(LSTM(units=50, return_sequences = False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer="adam", loss = "mean_squared_error")
    model.fit(x_train, y_train, epochs=1, batch_size=32)

    return model

def test_for_tomorrow(company, predict_length, model,scaler):
    today = datetime.now()
    start = dt.datetime(2012,1,1)
    yfin.pdr_override()
    data = pdr.get_data_yahoo(company, start, today)
    df = data.filter(["Close"])

    last_x_days = df[-predict_length:].values
    last_x_days_scaled = scaler.transform(last_x_days)
    x_test = []

    x_test.append(last_x_days_scaled)
    x_test = np.array(x_test)

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
    predicted_price = model.predict(x_test)

    predicted_price = scaler.inverse_transform(predicted_price)
    print(predicted_price)
    return predicted_price

def guess_prices(predict_length, model,data,scaler):
    x_test = []
    y_test = []
    for x in range(60, len(data)):
        x_test.append(data[x-predict_length:x,0])
        y_test.append(data[x,0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    return predicted_prices, y_test


def determine_weights(ps1,ps2,ps3, y_test):
    price_matrix = []
    price_matrix = np.concatenate((np.array(ps1), np.array(ps2), np.array(ps3)), axis=1)

    x_train, y_train = np.array(price_matrix), np.array(y_test)
    # x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
    
    model = LinearRegression()
    model.fit(x_train,y_train)

    return model

company = "TSLA"

today = datetime.now()

start = dt.datetime(2012,1,1)
end = today - timedelta(days=100)
yfin.pdr_override()
data = pdr.get_data_yahoo(company, start, end)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1,1))

long_predict = 60
med_predict = 30
short_predict = 7

model1 = build_model(long_predict, scaled_data)
model2 = build_model(med_predict, scaled_data)
model3 = build_model(short_predict, scaled_data)

p1 = test_for_tomorrow(company, long_predict, model1)
p2 = test_for_tomorrow(company, med_predict, model2)
p3 = test_for_tomorrow(company, short_predict, model3)

ps1, y = guess_prices(long_predict, model1, scaled_data)
ps2, y = guess_prices(med_predict, model2, scaled_data)
ps3, y = guess_prices(short_predict , model3, scaled_data)

stacked = determine_weights(ps1, ps2,ps3,y)

predicted_price = stacked.predict(np.concatenate((np.array(p1),np.array(p2),np.array(p3)),axis=1))
predicted_price = predicted_price.reshape(-1,1)
predicted_price = scaler.inverse_transform(predicted_price)
print(predicted_price)