import numpy as np
import pandas as pd
from scipy.stats import norm

#%% Def Function
def getSNInfo(SNType):
    #['side', 'option1', 'option2', option1_sign, option2_sign ,SBL(Boolean), 'settlement']
    if SNType == 'BULLPTS':
        return ['BULL', 'P', 'P', 1, -1, 0, 'Stock']
    
    elif SNType == 'BULLPTC':
        return ['BULL', 'P', 'P', 1, -1, 0, 'Cash']
    
    elif SNType == 'BULLNPS':
        return ['BULL', 'P', 'P', 1, 0, 0, 'Stock']
    
    elif SNType == 'BULLNPC':
        return ['BULL', 'P', 'P', 1, 0, 0, 'Cash']
    
    elif SNType == 'BEARPTC':
        return ['BEAR', 'C', 'C', 1, -1, 1, 'Cash']
    
    elif SNType == 'BEARNPC':
        return ['BEAR', 'C', 'C', 1, 0, 1, 'Cash']

def getBondPrice(calDay, intRate=0.0137, DayinYr=365):
    return 1 / (1 + (intRate * (calDay - 1) / DayinYr))

def getYield(BondPrice, calDay, DayinYr=365):
    return (1 / BondPrice - 1) * (DayinYr / calDay)

def getd(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return [d1, d2]

def getBSPrice(S, K, T, r, sigma, flag):
    N = norm.cdf
    d1, d2 = getd(S, K, T, r, sigma)
    
    if flag == 'C':
        return S * N(d1) - K * np.exp(-r*T) * N(d2)
    elif flag == 'P':
        return K * np.exp(-r*T) * N(-d2) - S * N(-d1)
    
#%% Input

## Bond ##
SNType = 'BULLNPC'
calDay = 35
intRate = 0.0137
DayinYr = 365
OptionDay = 35
BusDayinYr = 250

## Option ##
S = 100.
StrikeLvl = 0.98
K = 100 * 0.98
T = OptionDay / BusDayinYr
r = 0.025
sigma = 0.2
flag = 'P'

## SN ##
Notional = 1.

#%% Get SN Info
SNInfo = getSNInfo(SNType)
print(SNInfo)

#%% Get Bond Price
BondPrice = getBondPrice(calDay, intRate, DayinYr)
print(BondPrice)

#%% Get Option Price
OptionPrice = getBSPrice(S, K, T, r, sigma, flag)
OptionUnit = np.floor(Notional * (10**6) / K / 100) * 100
print(OptionUnit)

#%% Get SNPrice
SNPrice = (BondPrice - SNInfo[3] * OptionPrice / K)
print(SNPrice)

#%% Get Yield
Yield = getYield(SNPrice, calDay, DayinYr)
print(Yield)

#%% Get % Note
Note = SNPrice / 1 * 100
print(Note)

