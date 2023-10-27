
import numpy as np
import matplotlib.pyplot as plt


from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

class StackerModel:
    def __init__(self, model1, model2, model3):
        self.look_ahead = model1.look_ahead
        self.prices1 = model1.predicted_prices
        self.prices2 = model2.predicted_prices
        self.prices3 = model3.predicted_prices
        self.labels = model1.labels
        self.real_prices = model1.raw_data["Close"].values[-20:]
        
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.yscaler = MinMaxScaler(feature_range=(0,1))
        self.model = self.determine_weights()
        
        self.price_matrix = None
        self.tmr_price = self.weighted_price(model1.tmr_price,model2.tmr_price,model3.tmr_price,)
        self.weighted_prices = self.weighted_price(self.prices1,self.prices2, self.prices3)
        self.plot_price = np.concatenate([self.tmr_price])
        rp = self.real_prices
        self.p1 = model1.tmr_price
        self.p2 = model2.tmr_price
        self.p3 = model3.tmr_price

        # print(self.p1)
        # print(self.p2)
        # print(self.p3)
        # print(self.real_prices)
    
    
    def determine_weights(self):
        price_matrix = []
        # p1 = np.array(self.prices1).reshape
        price_matrix = np.concatenate((np.array(self.prices1), np.array(self.prices2), np.array(self.prices3)),axis=1)
        price_matrix = price_matrix.reshape(-1,60 )
        scaled_x_data = self.scaler.fit_transform(price_matrix)
        scaled_y_data = self.yscaler.fit_transform(self.labels)
        
        x_train, y_train = np.array(scaled_x_data), np.squeeze(np.array(self.labels))
        # x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
        # y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1],1))
        model = self.build_model()
        model.compile(optimizer="adam", loss = "mean_squared_error")
        model.fit(x_train, y_train, epochs=1, batch_size=32)
        return model

    def build_model(self):
        model = Sequential()
        # model.add(LSTM(units=50, return_sequences = True, input_shape=(x_train.shape[1],1)))
        model.add(Dense(units = 60))
        model.add(Dense(units = 20))
        return model

    def weighted_price(self, p1,p2,p3):
        input = np.concatenate((p1,p2,p3),axis=1)
        # input = np.squeeze(input)
        scaled_input = self.scaler.transform(input)
        # y_scale = self.yscaler.fit_transform()
        # scaled_input = np.reshape(scaled_input, (scaled_input[0], scaled_input[1],1))
        predicted_price = self.model.predict(scaled_input)
       
    
        predicted_price = self.yscaler.inverse_transform(predicted_price)
        return predicted_price
    
    def plot_results(self):
        plt.plot(np.squeeze(self.p1), color = "blue")
        plt.plot(np.squeeze(self.p2), color = "orange")
        plt.plot(np.squeeze(self.p3), color = "green")
        plt.plot(self.real_prices, color = "black")
        # plt.plot(self.plot_price, color= "red")
        plt.show()

    def get_accuracy(self): 
        diffs = np.absolute((self.real_prices- self.weighted_prices.flatten())/self.real_prices)
        average_error = np.average(diffs) 
        print(average_error)
       
           
    
