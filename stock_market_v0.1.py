#We determine our buy and sell points
import investpy
from pylab import *

def buy_sell(signal):
    buy=[]
    sell=[]
    flag = -1
    for i in range (0, len(signal)):
        if signal["MACD"][i] > signal["Signal"][i] :
            sell.append(np.nan)
            if flag != 1:
                buy.append(signal["Close"][i])
                flag= 1
            else:
                buy.append(np.nan)
        elif signal["MACD"][i] < signal["Signal"][i]:
            buy.append(np.nan)
            if flag !=0:
                sell.append(signal["Close"][i])
                flag = 0
            else:
                sell.append(np.nan)
        else:
            buy.append(np.nan)
            sell.append(np.nan)       
    return(buy,sell)

hisse="GEREL" # Write share
endeks="BIST 100"
dateilk="01/01/2019" #datefirst
dateson="26/08/2020" #datelast
df2 = investpy.get_index_historical_data(index=endeks, country="Turkey", from_date=dateilk, to_date=dateson)
df = investpy.get_stock_historical_data(stock=hisse, country="Turkey", from_date=dateilk, to_date=dateson)
#df2 = investpy.get_stock_historical_data(stock=hisse, country="Turkey", from_date=dateilk, to_date=dateson)
shortEMA = df.Close.ewm(span=30,adjust=False).mean()
longEMA = df.Close.ewm(span=48,adjust=False).mean()
MACD=shortEMA-longEMA
sinyal=MACD.ewm(span=10,adjust=False).mean()
df["MACD"]=MACD
df["Signal"]=sinyal
a = buy_sell(df)
df["Buy_signal"]=a[0]
df["Sell_signal"]=a[1]

shortEMA = df2.Close.ewm(span=30,adjust=False).mean()
longEMA = df2.Close.ewm(span=48,adjust=False).mean()
MACD=shortEMA-longEMA
sinyal=MACD.ewm(span=10,adjust=False).mean()
df2["MACD"]=MACD
df2["Signal"]=sinyal
b = buy_sell(df2)
df2["Buy_signal"]=b[0]
df2["Sell_signal"]=b[1]


plt.figure(figsize=(20,10))
plt.grid()
subplot(2,1,1)
plt.scatter(df.index,df["Buy_signal"],color="green",label="al",marker="*",alpha=1)
plt.scatter(df.index,df["Sell_signal"],color="red",label="sat",marker="*",alpha=1)
plt.plot(df["Close"],label="Kapanış",alpha=0.5)
plt.title(hisse)
plt.ylabel("PUAN")
plt.grid()

subplot(2,1,2)
plt.scatter(df2.index,df2["Buy_signal"],color="green",label="al",marker="*",alpha=1)
plt.scatter(df2.index,df2["Sell_signal"],color="red",label="sat",marker="*",alpha=1)
plt.plot(df2["Close"],label="Kapanış",alpha=0.5)
plt.title(endeks)
plt.ylabel("PUAN")
plt.xlabel("TARIH")
plt.grid()
plt.show()

