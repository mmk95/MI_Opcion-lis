from datetime import datetime, timedelta
import yfinance as yf
from keras import Sequential
from keras.layers import Dense, Dropout, LSTM
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler 
import numpy as np 
 


nvidia = "NVDA"
get_data =  yf.Ticker(nvidia)
test_end = datetime.now()
end_date = test_end - timedelta(days=1)
end_date_str = end_date.strftime("%Y-%m-%d")
data = get_data.history(start="2018-01-01", end=end_date_str)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
prediction_days =60

x_train = [] 
y_train = [] 

for i in range(prediction_days, len(scaled_data)): 
    x_train.append(scaled_data[i-prediction_days:i, 0]) 
    y_train.append(scaled_data[i, 0]) 

x_train, y_train = np.array(x_train), np.array(y_train) 
X_train = np.reshape(x_train, 
                     (x_train.shape[0], 
                      x_train.shape[1], 1))

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape =(X_train.shape[1], 1)))
model.add(Dropout(0.2)) 
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2)) 
model.add(LSTM(units=50))
model.add(Dropout(0.2)) 
model.add(Dense(units=1)) 


model.compile(optimizer='adam', loss = 'mean_squared_error')
model.fit(x_train, y_train, epochs = 25, batch_size = 32)

test_end = datetime.now()
end_date = test_end - timedelta(days=1)
end_date_str = end_date.strftime("%Y-%m-%d")

test_data = get_data.history(start="2024-01-01", end=end_date_str)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset)-len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)

x_test = []

for i in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[i-prediction_days:i,0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler. inverse_transform(predicted_prices)

plt.plot(actual_prices, color = 'black', label = f'Actual')
plt.plot(predicted_prices, color = 'green', label = f'Predicted')
plt.title('NVidia Stocks')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs + 1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data  , (real_data.shape[0], real_data.shape[1],1))
prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f'Predicted value = {prediction}')

