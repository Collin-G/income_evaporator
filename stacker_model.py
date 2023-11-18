
import numpy as np
import matplotlib.pyplot as plt


from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Flatten
from sklearn.ensemble import RandomForestRegressor

class StackerModel:
    def __init__(self, model1, model2, model3, data, scaler):
        self.look_ahead = model1.look_ahead
        self.scaler = scaler
        self.data = data
        self.prices1 = model1.predicted_prices
        self.prices2 = model2.predicted_prices
        self.prices3 = model3.predicted_prices
        self.labels = model1.labels
        self.real_prices = data
        
        self.model = self.determine_weights()
        
        self.price_matrix = None
        self.tmr_price = self.weighted_price(model1.tmr_price,model2.tmr_price,model3.tmr_price,)
        self.weighted_prices = self.weighted_price(self.prices1,self.prices2, self.prices3)
        self.plot_price = np.concatenate([self.tmr_price])
        
        rp = self.real_prices
        self.p1 = model1.tmr_price
        self.p2 = model2.tmr_price
        self.p3 = model3.tmr_price
    
    def determine_weights(self):
        price_matrix = []
       
        price_matrix = np.concatenate((np.array(self.prices1), np.array(self.prices2), np.array(self.prices3)),axis=1)
        x_train, y_train = np.array(price_matrix), np.squeeze(np.array(self.labels))
        model = self.build_model()
        model = model.fit(x_train, y_train)
        return model

    def build_model(self):
        # model = RandomForestRegressor(n_estimators=10)
        model = LinearRegression()
        # model.add(Flatten(input_shape=(3,20)))
        return model

    def weighted_price(self, p1,p2,p3):
        input = np.concatenate((p1,p2,p3),axis=1)
        predicted_price = self.model.predict(input)
       
    
        predicted_price = self.scaler.inverse_transform(predicted_price)
        return predicted_price
    
    def plot_results(self):
        plt.plot(np.squeeze(self.scaler.inverse_transform(self.real_prices[-20:])), color = "black")
        plt.plot(np.squeeze(self.tmr_price), color= "red")
        plt.show()

    def get_accuracy(self): 
        diffs = np.absolute((self.real_prices- self.weighted_prices.flatten())/self.real_prices)
        average_error = np.average(diffs) 
        print(average_error)
       
           
    
