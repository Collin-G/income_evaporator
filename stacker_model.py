
import numpy as np
import matplotlib.pyplot as plt


from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression


class StackerModel:
    def __init__(self, model1, model2, model3):
        self.prices1 = model1.predicted_prices
        self.prices2 = model2.predicted_prices
        self.prices3 = model3.predicted_prices
        self.labels = model1.labels
        self.real_prices = model1.raw_data["Close"].values[60:]
        self.scaler = model1.scaler
        self.model = self.determine_weights()
        self.tmr_price = self.weighted_price(model1.tmr_price,model2.tmr_price,model3.tmr_price)
        self.weighted_prices = self.weighted_price(self.prices1,self.prices2, self.prices3)
        
    def determine_weights(self):
        price_matrix = []
        price_matrix = np.concatenate((np.array(self.prices1), np.array(self.prices2), np.array(self.prices3)), axis=1)

        x_train, y_train = np.array(price_matrix), np.array(self.labels)
        
        model = LinearRegression()
        model.fit(x_train,y_train)

        return model

    def weighted_price(self, p1,p2,p3):
        predicted_price = self.model.predict(np.concatenate((np.array(p1),np.array(p2),np.array(p3)),axis=1))
        predicted_price = predicted_price.reshape(-1,1)
        predicted_price = self.scaler.inverse_transform(predicted_price)
        return predicted_price
    
    def plot_results(self):
        plt.plot(self.real_prices, color = "black")
        plt.plot(self.prices1, color = "green")
        plt.plot(self.prices2, color="orange")
        plt.plot(self.prices3, color = "blue")
        plt.plot(self.weighted_prices, color= "red")
        plt.show()

    def get_accuracy(self): 
        diffs = np.absolute((self.real_prices-  self.weighted_prices.flatten()/self.real_prices))
        average_error = np.average(diffs) 
        print(average_error)
       
           
    
