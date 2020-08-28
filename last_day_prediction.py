import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import investpy


hisse = "PETKM"  # Write share
dateilk = "01/01/2017" #datefirst
dateson = "18/08/2020" #datelast
datatahmin="19/08/2020" #date_prediction
df = investpy.get_stock_historical_data(stock=hisse, country="Turkey", from_date=dateilk, to_date=dateson)

data = df.filter(["Close"])
dataset = data.values
training_data_len = math.ceil(len(dataset) * .8)
# print(training_data_len)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
# print(scaled_data)
training_data = scaled_data[0:training_data_len, :]
x_train = []
y_train = []

for i in range(60, len(training_data)):
    x_train.append(training_data[i - 60:i, 0])
    y_train.append(training_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=1)
test_data = scaled_data[training_data_len - 60:, :]
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i - 60:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
train = data[:training_data_len]
valid = data[training_data_len:]
valid["Predictions"] = predictions
plt.figure(figsize=(16, 8))
plt.title(hisse)
plt.xlabel("Date")
plt.ylabel("Fiyat")
plt.plot(train["Close"])
plt.plot(valid[["Close", "Predictions"]])
plt.legend(["Train", "Val", "Predictions"], loc="lower right")


df2 = investpy.get_stock_historical_data(stock=hisse, country="Turkey", from_date=dateilk, to_date=dateson)
new_df=df2.filter(["Close"])
last60=new_df[-60:].values
last60_scaled=scaler.transform(last60)
X_test = []
X_test.append(last60_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
pred_price=model.predict(X_test)
pred_price=scaler.inverse_transform(pred_price)
print(pred_price)
df = investpy.get_stock_historical_data(stock=hisse, country="Turkey", from_date=dateson, to_date=datatahmin)
print(df["Close"])
plt.show()