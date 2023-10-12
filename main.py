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

from model import Model
from stacker_model import StackerModel
company = "TSLA"

long_predict = 60
med_predict = 30
short_predict = 7

model1 = Model(company, long_predict)
model2 = Model(company, med_predict)
model3 = Model(company, short_predict)


stacked = StackerModel(model1, model2,model3)
