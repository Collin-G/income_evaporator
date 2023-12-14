
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from pandas_datareader import data as pdr
import datetime as dt
import tensorflow.keras.backend as K
from datetime import datetime, timedelta

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Reshape, concatenate, Input, Lambda, Flatten, Activation
import yfinance as yfin

class Model():

    def __init__(self, data, scaler, predict_length, longest_predict_length,look_ahead):
        self.raw_data = None
        self.look_ahead = look_ahead
        self.scaler, self.data = scaler, data
        self.predict_length = predict_length
        self.longest_predict_length = longest_predict_length
        self.model = self.build_model()
        self.predicted_prices, self.labels = self.guess_prices2()
        self.tmr_price = self.future_projection2() #self.future_projection(50)

    def train_2(self, predict_length, look_ahead):
        x_train = []
        y_train = []
        data = self.data[:-100]

        for i in range(self.longest_predict_length, len(data)):
            if look_ahead+i >= len(data):
                break
            x_train.append(data[i-predict_length:i,0])
            y_train.append(data[i:i+look_ahead,0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
        y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1],1))
        return x_train, y_train

    def build_model(self):

        x_train, y_train = self.train_2(self.predict_length, self.look_ahead)
        model = Sequential()
        model.add(LSTM(units=70, return_sequences = True, input_shape=(x_train.shape[1],1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=35, return_sequences = False))
        model.add(Dense(units=20))
        model.add(Dropout(0.2))
        model.add(Dense(units=self.look_ahead))

        model.compile(optimizer="adam", loss = "mean_squared_error")
        model.fit(x_train, y_train, epochs=1, batch_size=32)
        summary = model.summary()
        print(summary)
        return model
    # def refine
    def build_functional_model(self):
        input = tf.keras.Input(shape=(1, self.predict_length))

        lstm1 = LSTM(units=60, return_sequences = True, input_shape=(input.shape[1],1))
        drop1 = Dropout(0.2)(lstm1)
        lstm2 = LSTM(units=35, return_sequences = False)(drop1)
        dense1 = Dense(units=30)(lstm2)
        drop2 = Dropout(0.2)(dense1)
        dense2 = Dense(units = 1)(drop2)

        # Represents a copy of model:
        single_day_model = dense2

        #inputs = []
        #inputs = np.array(input)
        inputs = input
        outputs = [single_day_model(input)]
        #inputs = inputs.eval()
        for i in range(self.look_ahead):
            cur_layer = concatenate(inputs[i:], outputs)
            #cur_layer = concatenate(outputs[i-1], )
            #inputs= inputs.append(out)
            #inputs = np.delete(inputs,0)
            #inputs = K.constant(inputs)


        outputs = inputs

        model = tf.keras.Model(inputs = input, outputs = outputs, name = "test")
        X_train, Y_train = self.train_2(self.predict_length, self.look_ahead)
        model.fit(X_train, Y_train, epochs=1, batch_size=32)
        return model
    
    def build_functional_long_long_model(self):
        input = tf.keras.Input(shape=(self.predict_length, 1))

        class SingleDayModelInstance(keras.layers.Layer):
            def __init__(self):
                self.lstm1 = LSTM(units=60, return_sequences = True, input_shape=(input.shape))
                self.lstm2 = LSTM(units=35, return_sequences = False)
                self.dense1 = Dense(units=30)
                self.drop2 = Dropout(0.2)
                self.dense2 = Dense(units = 1)
                self.reshape1 = Reshape((1, 1))
                super().__init__()
                
            def call(self, input):
                x = self.lstm1(input)
                x = self.lstm2(x)
                x = self.dense1(x)
                x = self.drop2(x)
                x = self.dense2(x)
                x = self.reshape1(x)

                return x

        # Represents a copy of model:
        single_day_model = SingleDayModelInstance()
        outputs_raw = single_day_model(input)

        for i in range(1, self.look_ahead):
            concatted = concatenate([input, outputs_raw], axis=1)
            cur_input = concatted[i:i+self.predict_length]
            outputs_raw = concatenate([outputs_raw, single_day_model(cur_input)], axis=1)

        # Reshape outputs
        outputs = Reshape((-1,))(outputs_raw)

        model = tf.keras.Model(inputs = input, outputs = outputs, name = "test")

        X_train, Y_train = self.train_2(self.predict_length, self.look_ahead)
        model.fit(X_train, Y_train, epochs=1, batch_size=32)
        return model


    def future_projection2(self):
        data = self.data[:-self.look_ahead]
        prices = self.test_for_tomorrow(data[-self.predict_length:])
        return prices  

    def test_for_tomorrow(self, data):
        x_train = []
        x_train.append(data)
        x_train = np.array(x_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
        predicted_price = self.model.predict(x_train)
        return predicted_price

    def guess_prices2(self):
        x_test = []
        y_test = []
        for x in range(self.longest_predict_length, len(self.data)):
            if self.look_ahead+x >= len(self.data):
                break
            x_test.append(self.data[x-self.predict_length:x,0])
            y_test.append(self.data[x:x+self.look_ahead,0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
        y_test = np.array(y_test)
        y_test = np.reshape(y_test, (y_test.shape[0],y_test.shape[1],1))

        predicted_prices = self.model.predict(x_test)
        return predicted_prices, y_test
   
            
