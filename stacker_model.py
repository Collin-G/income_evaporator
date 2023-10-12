
import numpy as np
import matplotlib.pyplot as plt


from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression


class StackerModel:
    def __init__(self, prices1, prices2, prices3, labels):
        self.prices1 = prices1
        self.prices2 = prices2
        self.prices3 = prices3
        self.labels = labels
        self.scaler = MinMaxScaler(feature_range=(0,1))

        self.model = self.determine_weights()
        
    def determine_weights(self):
        price_matrix = []
        price_matrix = np.concatenate((np.array(self.prices1), np.array(self.prices2), np.array(self.prices3)), axis=1)

        x_train, y_train = np.array(price_matrix), np.array(self.labels)
        
        model = LinearRegression()
        model.fit(x_train,y_train)

        return model

    def weighted_price(self):
        predicted_price = self.model.predict(np.concatenate((np.array(self.prices1),np.array(self.prices2),np.array(self.prices3)),axis=1))
        predicted_price = predicted_price.reshape(-1,1)
        predicted_price = self.scaler.inverse_transform(predicted_price)
        return predicted_price
    
