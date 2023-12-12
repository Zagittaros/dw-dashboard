import pandas as pd
import numpy as np
from datetime import datetime, timedelta

file_name = 'RTDPrice.xlsm'
sheet_name = 'Price'

data = pd.read_excel(file_name, sheet_name) 

df = data.copy()
df = df.sort_values(['Idx'], ascending=False)
df = df.replace('retrieving', np.nan)
df.drop('Idx', axis=1, inplace=True)
df.set_index('Date', inplace=True)

def CalReturn(Price_data, MaxDays=189):
    cap = np.minimum((len(Price_data) - MaxDays), 250)
    NewPrice_data = Price_data.iloc[-cap:]
    LongReturn = (NewPrice_data / NewPrice_data.shift()) - 1
    return LongReturn

def RealVol(Price_data, window=30, tradingDays=250):
    LongReturn = CalReturn(Price_data)
    RealizedVol = LongReturn.rolling(window=window).std() * np.sqrt(tradingDays)
    return RealizedVol

def VolCone(Price_data, window, p, MaxDays=189):
    
    ## Calculate LongReturn ##
    LongReturn = CalReturn(Price_data, MaxDays)
    
    ## Calculate Vol ##
    CountRow = np.minimum((len(Price_data) - window), 250)
    
    ReturnObs = LongReturn.iloc[-(CountRow + window - 1):]
    LongVol = ReturnObs.rolling(window=window).std() * np.sqrt(tradingDays)
    Vol_at_p = np.nanpercentile(LongVol, p)
    return Vol_at_p

Price_data = df
windows = [x for x in range(21, 190, 21)]
mid_p = 50
bottom_p = 3
tradingDays = 250
MaxDays = windows[-1]

tickers = df.columns
RealizedVol = np.zeros([len(tickers), len(windows)])
MeanVol = np.zeros([len(tickers), len(windows)])

VolCone50 = np.zeros([len(tickers), len(windows)])
VolCone3 = np.zeros([len(tickers), len(windows)])

## Calculate Vol ##
for i in range(len(tickers)):
    for j in range(len(windows)):
                    
        estimator = RealVol(Price_data[tickers[i]], windows[j], tradingDays)
        
        ## Realized Vol ##
        RealizedVol[i][j] = estimator[-1]
        
        ## Average Vol ##
        MeanVol[i][j] = np.mean(RealizedVol[i][:j+1])
        
        ## 50th Percentile Vol ##
        VolCone50[i][j] = VolCone(Price_data[tickers[i]], windows[j], mid_p, MaxDays)
        VolCone3[i][j] = VolCone(Price_data[tickers[i]], windows[j], bottom_p, MaxDays)

AdjFactor = pd.read_excel(file_name, sheet_name='F1', header=None)

F1 = np.array(AdjFactor.iloc[0])
F2 = 1.4

BuyVol = np.zeros([len(tickers), len(windows)])
HedgeVol = np.zeros([len(tickers), len(windows)])
SellVol = np.zeros([len(tickers), len(windows)])

for i in range(len(tickers)):
    for j in range(len(windows)):
        
        MidVol = VolCone50[i][j]
        BottVol = VolCone3[i][j]
        AvgVol = MeanVol[i][j]
        
        BuyVol[i][j] = np.maximum(((np.minimum(MidVol, AvgVol)) + 0.25 * np.absolute(MidVol-AvgVol)) * F1[i], BottVol * F1[i])
        HedgeVol[i][j] = np.maximum(MidVol, AvgVol)
        SellVol[i][j] = ((np.maximum(MidVol, AvgVol)) - 0.25 * np.absolute(MidVol-AvgVol)) * F2

BuyVol_df = pd.DataFrame(BuyVol, index=tickers, columns=windows)
SellVol_df = pd.DataFrame(SellVol, index=tickers, columns=windows)
HedgeVol_df = pd.DataFrame(HedgeVol, index=tickers, columns=windows)

VolDB = pd.concat([BuyVol_df, SellVol_df, HedgeVol_df])
VolDB.index.names = ['Underlying']
VolDB.reset_index(inplace=True)

VolDB.to_excel('VolDB.xlsx')