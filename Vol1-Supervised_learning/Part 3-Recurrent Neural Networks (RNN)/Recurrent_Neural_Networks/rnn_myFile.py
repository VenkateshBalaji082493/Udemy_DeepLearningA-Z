# Recurrent Neural Network LSTM to predict Stock trend of Google

# Part1- Data Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')

#we get only the 2nd column of the spreadsheet but this is done to create
#numpy array
training_set = dataset_train.iloc[:,1:2].values

# Feature scaling (to ease the training process)
# normalization is used because sigmoid activation function is used                                         

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))

# fit gets the min and max of the data; transform does the transformation
training_set_scaled = sc.fit_transform(training_set)

# Creating data structure with 60 timesteps and 1 output

X_train = []
y_train = []

for i in range(60,1258):
    X_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])
X_train,y_train=np.array(X_train),np.array(y_train) # making it numpy array

# Reshaping
# if there are several indicators last arg varies
# this reshaping is done to get D tensor inpur to the RNN
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) 
    
# Part2- Building RNN

# Importing libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initializing RNN
regressor= Sequential() #RNN

neurons=50
output=1
# Adding the first LSTM layer and dropout Regularization 
regressor.add(LSTM(units = neurons, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(rate=0.2)) # 10 neurons are ignored in each iteration

# Adding the second LSTM layer and dropout Regularization
 # no need for input_shape as the units=50 in previous layer
regressor.add(LSTM(units = neurons, return_sequences=True))
regressor.add(Dropout(rate=0.2))

# Adding the third LSTM layer and dropout Regularization
regressor.add(LSTM(units = neurons, return_sequences=True))
regressor.add(Dropout(rate=0.2))

# Adding the fourth LSTM layer and dropout Regularization
regressor.add(LSTM(units = neurons, return_sequences=False))
regressor.add(Dropout(rate=0.2))

# Adding the output layer
regressor.add(Dense(units=1)) # fully connected layer

# Compiling the RNN
regressor.compile(optimizer='adam',loss='mean_squared_error')

# Training the RNN
regressor.fit(X_train, y_train, epochs=100, batch_size= 32) # for every 32 input iterations, weights are updated

# Part3- Making the prediction and visualizing the results

# Getting the real stock price of 2017

dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')

real_stock_price = dataset_test.iloc[:,1:2].values

# Predicted stock price of Jan 2017
dataset_total=pd.concat((dataset_train['Open'],dataset_test['Open']),axis = 0)
inputs=dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs= inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test= []
for j in range(60,80):
    X_test.append(inputs[j-60:j,0])
X_test=np.array(X_test) # making it numpy array
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price=sc.inverse_transform(predicted_stock_price)
input_back=sc.inverse_transform(inputs)
# visualizing both data 
plt.plot(real_stock_price, color='red', label= 'Real Stock Price' )
plt.plot(predicted_stock_price, color='green', label= 'Predicted Stock Price')
plt.legend()
plt.title('Stock Price Prediction')
plt.xlabel('time')
plt.ylabel('Stock') 
plt.show()

import math
from sklearn.metrics import mean_squared_error
rmse=math.sqrt(mean_squared_error(real_stock_price,predicted_stock_price))