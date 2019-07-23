import pandas as pd
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def date_parser(x):
    return datetime.strftime(x, '%Y-%m-%d')

# date-time parsing function for loading the dataset
def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')
 
# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df
 
# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)
 
# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]
 
# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled
 
# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]
 
# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    #model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True, )
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True, return_sequences=True))
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    train_loss = list()
    val_loss = list()
    for i in range(nb_epoch):
        history = model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False, validation_split=0.33)
        train_loss.append(history.history['loss'])
	val_loss.append(history.history['val_loss'])
	model.reset_states()
    return model, train_loss, val_loss

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

#####################################
start = time.time()
# read data
series = pd.read_csv("../vsean_data/post_time_series.csv", header=0, parse_dates=[0], index_col=0, squeeze=True, names=["Date","Count"])
# series["Date"] = pd.to_datetime(series['Date'])
# series = series.set_index(pd.DatetimeIndex(series["Date"]))
print(series.head())
print("\n")

# transform data to be stationary
raw_values = series.values
diff_values = difference(raw_values, 1)
print(diff_values.head())
print("\n")

# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values
print(supervised.head())

#[-336:] test set contains 2018 data
#[-1067:-336] train set contains 2016-2017 data
# split data into train and test-sets
train, test = supervised_values[:-336], supervised_values[-336:]
print("train size: ", len(train), "\n test size: ", len(test))
print(train[:5])
print(test[:5])

# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)

# fit the model
nb_epochs = 300
lstm_model, train_loss, val_loss = fit_lstm(train_scaled, 1, nb_epochs, 4)

# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)

#*************
# date of test data
a=series.index.tolist()
a_date = [date_parser(x) for x in a]
day_test = a_date[-len(test):]
#************

# walk-forward validation on the test data
predictions = list()
for i in range(len(test_scaled)):
	# make one-step forecast
	X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
	yhat = forecast_lstm(lstm_model, 1, X)
	# invert scaling
	yhat = invert_scale(scaler, X, yhat)
	# invert differencing
	yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
	# store forecast
	predictions.append(yhat)
	expected = raw_values[-len(test) + i + 1]
	print('Date=%s, Predicted=%f, Expected=%f' % (day_test[i], yhat, expected))

# report performance
rmse = sqrt(mean_squared_error(raw_values[-336:], predictions))
print('Test RMSE: %.3f' % rmse)

data_used = "data_16-17"
fig1 = "lstm_result" + data_used + "_forecast_18.jpg"
fig2 = "lstm_loss_vs_epoch" + data_used + ".jpg"
file1 = "lstm_result" + data_used + "_forecast_18.csv"
file2 = "lstm_loss_vs_epoch" + data_used + ".csv"


# line plot of observed vs predicted
plt.plot(raw_values[-336:], label="truth")
plt.plot(predictions, label="forecast")
plt.title('LSTM Forecast')
plt.ylabel('Number of Activities')
plt.xlabel('Date')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2)
#plt.show()
plt.savefig('figures/'+fig1)
print("Figure saved! "+fig1)
plt.gcf().clear()

np.savetxt("results/"+file1, np.column_stack((raw_values[-336:].tolist(),predictions )), delimiter=",", fmt='%s', header="truth, predict")
print("Result saved! "+file1)

# print loss
plt.plot(train_loss, label="train")
plt.plot(val_loss, label="validate")
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2)
#plt.show()
plt.savefig("figures/"+fig2)
print("Figure saved! "+fig2)
plt.gcf().clear()

np.savetxt("results/"+file2, np.column_stack((train_loss, val_loss)), delimiter=",", fmt='%s', header="train_loss, val_loss")
print("Result saved! "+file2)

end = time.time()
print(end - start)
