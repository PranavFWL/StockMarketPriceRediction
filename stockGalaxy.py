import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import numpy as np

data = yf.download(tickers = '^NSEI', start = '2000-01-01', end = '2024-05-24')
col = data.columns
df = data.values
data = pd.DataFrame(df, columns= col)
fi_data = data[['Open',	'High','Low',	'Close']]

# Simple Moving Average (SMA)
fi_data['SMA5'] = ta.sma(fi_data['Close'], length=5)
fi_data['SMA10'] = ta.sma(fi_data['Close'], length=10)

# Exponential Moving Average (EMA)
fi_data['EMA5'] = ta.ema(fi_data['Close'], length=5)
fi_data['EMA10'] = ta.ema(fi_data['Close'], length=10)

fi_data['KAMA10'] = ta.kama(fi_data['Close'], length=10)

data = fi_data.iloc[10:]

target = pd.DataFrame()

target['1open'] = data['Open'].shift(-1)
#target['1high'] = data['High'].shift(-1)
#target['1low'] = data['Low'].shift(-1)
#target['1close'] = data['Close'].shift(-1)

target = target.iloc[:-1, :]
data = data.iloc[:-1,:]

x_len = int(len(data) * 0.8)

x_train_pre = data.head(x_len)
y_train_pre = target.head(x_len)

x_test_pre = data.iloc[x_len:, :]
y_test_pre = target.iloc[x_len:, :]

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))

x_train = sc.fit_transform(x_train_pre)
x_test = sc.fit_transform(x_test_pre)
y_train = sc.fit_transform(y_train_pre)
y_test = sc.fit_transform(y_test_pre)

from keras.layers import LSTM
from keras.layers import TimeDistributed, Dense, Input, Activation, concatenate

from keras import optimizers
from keras.models import Model

lstm_in = Input(shape=(9,1), name='lstm_in')
inputs = LSTM(150, name ='first_layer')(lstm_in)
inputs = Dense(1, name = 'dense_layer')(inputs)
output = Activation('linear', name= 'output')(inputs)
model = Model(inputs = lstm_in, outputs = output)
adam = optimizers.Adam()
model.compile(optimizer = adam, loss= 'mse')
model.fit(x = x_train, y = y_train, batch_size = 15, epochs = 30, shuffle = True, validation_split = 0.1)

prec = model.predict(x_test)

prediction = sc.inverse_transform(prec)
real = sc.inverse_transform(y_test)

real = np.vectorize(lambda x: int(x))(real)
prediction = np.vectorize(lambda x: int(x))(prediction)

real = real.flatten()
prediction = prediction.flatten()

plt.figure(figsize=(16,8))
plt.plot(prediction, color = 'black', label = 'Prediction')
plt.plot(real, color = 'red', label = 'Real')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

print('mean_squared_error', mean_squared_error(real, prediction))
print('mean_absolute_error',mean_absolute_error(real, prediction))
print('r2_score',r2_score(real, prediction))
print('explained_variance_score',explained_variance_score(real, prediction))

