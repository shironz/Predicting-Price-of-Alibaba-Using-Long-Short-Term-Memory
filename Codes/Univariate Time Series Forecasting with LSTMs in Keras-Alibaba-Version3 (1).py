
# coding: utf-8

# In[1]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout 
from keras.layers import LSTM


# In[2]:


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


# In[3]:


# now let get all the information for stock
stock =  read_csv('BABA20130425_20180425.csv', header=0)
print(stock.shape)
print(stock.head())


# In[4]:


# now get the stock close price
# astype means "Copy of the array, cast to a specified type."
stock_prices = stock.Close.values.astype("float32")
shape0=stock_prices.shape[0]
stock_prices = stock_prices.reshape(shape0, 1)
print(stock_prices.shape)
# print the prices of last five observations
print(stock_prices[-5:])


# In[5]:


# Before doing any analysis, first plot the prices series(data)
pyplot.plot(stock_prices)
pyplot.title('Stock close price over time')
pyplot.ylabel('Price')
pyplot.xlabel('Time Range')
pyplot.show()


# In[6]:


values = stock_prices


# In[7]:


# check the last five observations to make sure it's correct
print(values[-5:, :])


# In[8]:


# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
print(reframed.head())
# var1 means prices


# In[9]:


# split into train and test sets
values = reframed.values
n_train = int(reframed.shape[0] * 0.8)
train = values[:n_train, :]
test = values[n_train:, :]

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# In[10]:


# design network
model = Sequential()

model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(1))
model.add(Activation('linear'))

model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=300, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)


# In[11]:


# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


# In[12]:


# make a prediction for train data
yhat_train = model.predict(train_X)
train_X = train_X.reshape((train_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat_train = concatenate((yhat_train, train_X[:, 1:]), axis=1)
inv_yhat_train = scaler.inverse_transform(inv_yhat_train)
inv_yhat_train = inv_yhat_train[:,0]
# invert scaling for actual
train_y = train_y.reshape((len(train_y), 1))
inv_y_train = concatenate((train_y, train_X[:, 1:]), axis=1)
inv_y_train = scaler.inverse_transform(inv_y_train)
inv_y_train = inv_y_train[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y_train, inv_yhat_train))
print('Train RMSE: %.3f' % rmse)


# In[13]:


# plot only the train data and predicted train data
trainline, =pyplot.plot(inv_y_train, label='train data')  # blue one
trainPredictline, =pyplot.plot(inv_yhat_train, label='predicted train data') # orange one
pyplot.title("Train data perormance")
pyplot.ylabel('Stock price')
pyplot.xlabel('Time range')
pyplot.legend(handles=[trainline, trainPredictline])
pyplot.show()


# In[14]:


# plot only the last 100 train data and predicted train data
trainline, =pyplot.plot(inv_y_train[-30:], label='train data')  # blue one
trainPredictline, =pyplot.plot(inv_yhat_train[-30:], label='predicted train data') # orange one
pyplot.title("Train data perormance over recent 30 observations")
pyplot.ylabel('Stock price')
pyplot.xlabel('Time range')
pyplot.legend(handles=[trainline, trainPredictline])
pyplot.show()


# In[15]:


# check some observations of train data
print(inv_y_train.shape)
print(inv_yhat_train.shape)
# check the performance
print(inv_y_train[-1])
print(inv_yhat_train[-1])
#
print(inv_y_train[-2])
print(inv_yhat_train[-2])
#
print(inv_y_train[-3])
print(inv_yhat_train[-3])
#
print(inv_y_train[-4])
print(inv_yhat_train[-4])


# In[16]:


# plot the difference for train data
traindiff = inv_yhat_train - inv_y_train
#
maxtraindiff=abs(max(traindiff, key=abs))
print('The largest absolute difference for train data: %.2f' % (maxtraindiff))
#
pyplot.plot(traindiff)
#
pyplot.title("Predicted train data minus train data")
pyplot.ylabel('Difference')
pyplot.xlabel('Time range')
pyplot.show()


# In[17]:


# plot the difference for the recent 30 obs of train data
pyplot.plot(traindiff[-30:])
#
pyplot.title("Predicted train data minus train data over recent 30 observations")
pyplot.ylabel('Difference')
pyplot.xlabel('Time range')
pyplot.show()


# In[18]:


# make a prediction for test data
yhat_test = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat_test = concatenate((yhat_test, test_X[:, 1:]), axis=1)
inv_yhat_test = scaler.inverse_transform(inv_yhat_test)
inv_yhat_test = inv_yhat_test[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y_test = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y_test = scaler.inverse_transform(inv_y_test)
inv_y_test = inv_y_test[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y_test, inv_yhat_test))
print('Test RMSE: %.3f' % rmse)


# In[19]:


# plot only the test data and predicted test data
testline, =pyplot.plot(inv_y_test, label='test data')  # blue one
testPredictline, =pyplot.plot(inv_yhat_test, label='predicted test data') # orange one
pyplot.title("Test data perormance")
pyplot.ylabel('Stock price')
pyplot.xlabel('Time range')
pyplot.legend(handles=[testline, testPredictline])
pyplot.show()


# In[20]:


# plot only the last 30 test data and predicted test data
testline, =pyplot.plot(inv_y_test[-30:], label='test data')  # blue one
testPredictline, =pyplot.plot(inv_yhat_test[-30:], label='predicted test data') # orange one
pyplot.title("Test data perormance over recent 30 observations")
pyplot.ylabel('Stock price')
pyplot.xlabel('Time range')
pyplot.legend(handles=[testline, testPredictline])
pyplot.show()


# In[21]:


# check some observations of test data
print(inv_y_test.shape)
print(inv_yhat_test.shape)
# check the performance
print(inv_y_test[-1])
print(inv_yhat_test[-1])
#
print(inv_y_test[-2])
print(inv_yhat_test[-2])
#
print(inv_y_test[-3])
print(inv_yhat_test[-3])
#
print(inv_y_test[-4])
print(inv_yhat_test[-4])


# In[22]:


# plot the difference for test data
testdiff = inv_yhat_test - inv_y_test
#
maxtestdiff=abs(max(testdiff, key=abs))
print('The largest absolute difference for test data: %.2f' % (maxtestdiff))
#
pyplot.plot(testdiff)
#
pyplot.title("Predicted test data minus test data")
pyplot.ylabel('Difference')
pyplot.xlabel('Time range')
pyplot.show()


# In[23]:


# plot the difference for the recent 30 obs of test data
pyplot.plot(testdiff[-30:])
#
pyplot.title("Predicted test data minus test data over recent 30 observations")
pyplot.ylabel('Difference')
pyplot.xlabel('Time range')
pyplot.show()

