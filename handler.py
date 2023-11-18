from pandas_datareader import data as pdr
import datetime as dt
from datetime import datetime, timedelta

import yfinance as yfin

from sklearn.preprocessing import MinMaxScaler

from model import Model
from stacker_model import StackerModel

class Handler():
    def __init__(self,company, long_predict, med_predict, short_predict, look_ahead):
        self.scaler, self.data = self.build_data(company)
        model1 = Model(self.data,self.scaler, long_predict,long_predict, look_ahead)
        model2 = Model(self.data,self.scaler, med_predict,long_predict,look_ahead)
        model3 = Model(self.data, self.scaler,short_predict,long_predict,look_ahead)


        stacked = StackerModel(model1, model2,model3, self.data, self.scaler)
        stacked.plot_results()
    
    def build_data(self,company):
        today = datetime.now()

        start = today -timedelta(10000)
        end = today
        yfin.pdr_override()
        data = pdr.get_data_yahoo(company, start, end)
        self.raw_data = pdr.get_data_yahoo(company, start, today)
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1,1))
        return scaler,scaled_data
