import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

def getd(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma* np.sqrt(T)
    return d1, d2

def getDelta(S, K, T, r, sigma, d=0, flag='C'):
    N = norm.cdf
    d1, d2 = getd(S, K, T, r, sigma)
    
    if flag =='C':
        return N(d1)
    elif flag =='P':
        return N(d1) - 1
    
def getTheta(S, K, T, r, sigma, OptionDays=250, d=0, flag='C'):
    N = norm.cdf
    d1, d2 = getd(S, K, T, r, sigma)
    
    Nd1 = np.exp(-(d1**2)/2) / (np.sqrt(2 * np.pi))
    
    if flag =='C':
        return 1/OptionDays * -((S * sigma * Nd1) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * N(d2))
    elif flag == 'P':
        return 1/OptionDays * -((S * sigma * Nd1) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * (1 - N(d2)))
    
def getBSPrice(S, K, T, r, sigma, d=0, flag='C'):
    N = norm.cdf
    d1, d2 = getd(S, K, T, r, sigma)
    
    if flag == 'C':
        return S * N(d1) - K * np.exp(-r*T)* N(d2)
    elif flag == 'P':
        return K * np.exp(-r*T)* N(-d2) - S * N(-d1)
    
def getPayoff(S, K, flag='C'):
    if flag == 'C':
        return max(S-K, 0)
    elif flag == 'P':
        return max(K-S, 0)
    
def getSen(S, K, T, r, sigma, conv, tick=0.25, d=0, flag='C'):
    delta = getDelta(S, K, T, r, sigma, d, flag)
    return round((delta/conv) * (tick/0.01), 3)

def getGearing(S, K, T, r, sigma, conv, d=0, flag='C'):
    delta = getDelta(S, K, T, r, sigma, flag)
    Premium = getBSPrice(S, K, T, r, sigma, d, flag)
    return round(abs(delta) * (S/Premium), 3)

def getVega(S, K, T, r, sigma):
    N = norm.cdf
    d1, d2 = getd(S, K, T, r, sigma)
    Nd1 = np.exp(-(d1**2)/2) / (np.sqrt(2 * np.pi))
    return S * np.sqrt(T) * Nd1 * 0.01

def getGamma(S, K, T, r, sigma):
    d1, d2 = getd(S, K, T, r, sigma)
    Nd1 = np.exp(-(d1**2)/2) / (np.sqrt(2 * np.pi))
    return Nd1 / (S*sigma*np.sqrt(T))

## Fixed Parameters ##
DayinYr = 250
flag = 'C'
S = 44.25
sigma = 0.4 # ConeVol50 of the UL
r = 0.025
d = 0.
tick = 0.25

## Variable Parameters ##
Moneyness = np.arange(0.2, 0.4, 0.05)
TTM = np.around(np.arange(3.5, 5.5, 0.5) * 21)
Vol = np.arange(0.55, 0.7, 0.01)
conv = np.around(np.arange(1., 25.5, 0.1), 2)

Best_param = []
Result = []

## Parameter Search ##
for m in Moneyness:
    for t in TTM:
        for v in Vol:
            for c in conv:
                K = round(S * (1 + m) if flag == 'C' else S * (1 - m), 2)
                PricingVol = v
                
                Premium = round(getBSPrice(S, K, t/DayinYr, r, PricingVol, d, flag) / c, 2)
                DWSen = getSen(S, K, t/DayinYr, r, PricingVol, c, tick, d, flag)
                DWGear = getGearing(S, K, t/DayinYr, r, PricingVol, c, d, flag)
                DWDelta = round(getDelta(S, K, t/DayinYr, r, PricingVol, d, flag), 4)
                DWTheta = round(getTheta(S, K, t/DayinYr, r, PricingVol, 250, d, flag) / Premium * c, 4)
                
                Best_param.append([m, t, v, c])
                Result.append([K, PricingVol, Premium, DWSen, DWGear, DWDelta, DWTheta])
           
df = pd.DataFrame(np.concatenate([Best_param, Result], axis=1))
df.columns = ['Moneyness', 'TTM', 'MarkupVol', 'Conversion', 'Strike', 'PricingVol', 'Premium', 'Sensitivity', 'Gearing', 'Delta', 'Theta (%)']
df = df[(df.Sensitivity > 0.75) & (df.Sensitivity < 0.85)]
df = df[(df.Gearing > 6) & (df.Gearing < 10)]
df = df[df.Premium < 30]
df = df.nlargest(len(df), columns='Gearing')
df = df.nlargest(1, columns='Sensitivity')
print(df)