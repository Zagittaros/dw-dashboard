import numpy as np
import pandas as pd
from scipy.stats import norm
import plotly.express as px
# import streamlit as st

import plotly.io as pio
pio.renderers.default='browser' # browser, svg

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

def getGBM(S, T, mu, sigma, N, M):
    dt = T/N
    St = np.exp((mu - sigma ** 2 / 2) * dt + sigma * np.random.normal(0, np.sqrt(dt), size=(M,N)).T)
    St = np.vstack([np.ones(M), St])
    St = S * St.cumprod(axis=0)
    
    return St

## Sensitivity Simulation ##
def getSenProfile(S, K, T, r, sigma, conv, tick=0.25, d=0., flag='C', DayinYr=250, PriceRange=99, TimeSteps=1.):
    OptionDays = 250

    S_ = [round(x, 2) for x in np.arange(S-tick*PriceRange, S+tick*PriceRange, tick)]
    T_ = np.arange(T*DayinYr, 0, -TimeSteps)
    
    Sim = np.zeros([len(S_), len(T_)])

    for s in range(len(S_)):
        for t in range(len(T_)):
            Sim[s][t] = getSen(S_[s], K, T_[t]/DayinYr, r, sigma, conv, tick, d, flag)
            
    Sim_df = pd.DataFrame(Sim, columns=T_, index=S_).dropna().sort_index(ascending=False)
    
    min_ = np.min(Sim)
    max_ = np.max(Sim)+0.01

    UpperBound, LowerBound = [1.1, 0.5]

    if flag == 'C':
        colorscale = ['gray' if x > UpperBound else ('red' if x < LowerBound else 'green') for x in np.arange(min_, max_, 0.01)]
    else:
        colorscale = ['gray' if x < -UpperBound else ('red' if x > -LowerBound else 'green') for x in np.arange(min_, max_, 0.01)]
    
    fig = px.imshow(Sim_df, text_auto=False, aspect="auto", color_continuous_scale=colorscale, origin='lower',
                   title='DW Sensitivity Profile')

    fig.update_xaxes(autorange='reversed')

    fig.update_layout(font=dict(size=12,
                                 color='white'),
                      xaxis_title='Maturity',
                      yaxis_title='Spot Price',
                      hovermode="x unified"
                     )

    fig.update_layout({
        'plot_bgcolor': 'rgb(0,0,0,0)',
        'paper_bgcolor': '#222A2A'
    })

    fig.add_hline(y=S, line_color='yellow', line_dash="dash",
                  annotation_text=f"Spot = {S}", 
                  annotation_position="bottom right")

    fig.add_hline(y=K, line_color='orange', line_dash="dash",
                  annotation_text=f"Strike = {K}", 
                  annotation_position="bottom right")

    return fig

## Input ##
DayinYr = 250

flag = 'C'
S = 44.25
K = 59.70
d = 0.

r = 0.025
T = 74

PricingVol = 0.58
tick = 0.25
conv = 6.5

## Input Streamlit ##
# st.set_page_config(page_title='DW Dashboard',
#                    layout='wide')
# st.title('DW Simulation')

# st.sidebar.header('Input')
# with st.sidebar:
#     DayinYr = st.number_input("Days in a Year", value=250, min_value=250, placeholder="Type a number...")
#     flag = st.selectbox("Call/Put", ('C', 'P'))
#     S = st.number_input("Spot", value=70., min_value=1., placeholder="Type a number...")
#     K = st.number_input("Strike", value=75.5, min_value=1., placeholder="Type a number...")
#     T = st.number_input("TTM (Days)", value=118, min_value=1, placeholder="Type a number...")
#     PricingVol = st.number_input("Pricing Vol.", value=0.61, min_value=0., placeholder="Type a number...")
#     conv = st.number_input("Conversion Rate (DW:1 UL)", value=1., min_value=0.0001, placeholder="Type a number...")
#     tick = st.number_input("Tick Size", value=0.25, min_value=0.01, placeholder="Type a number...")
#     r = st.number_input("Discount Rate", value=0.025, min_value=0., placeholder="Type a number...")
#     d = st.number_input("Dividend Yield", value=0., min_value=0., placeholder="Type a number...")
#     PriceRange = st.number_input("Simulation Range", value=99, min_value=15, placeholder="Type a number...")
#     TimeSteps = st.number_input("Simulation Step", value=1, min_value=1, placeholder="Type a number...")

## Calculate Specification ##
Premium = getBSPrice(S, K, T/DayinYr, r, PricingVol, d, flag) / conv
DWSen = getSen(S, K, T/DayinYr, r, PricingVol, conv, tick, d, flag)
DWGear = getGearing(S, K, T/DayinYr, r, PricingVol, conv, d, flag)
DWDelta = getDelta(S, K, T/DayinYr, r, PricingVol, d, flag)
DWTheta = getTheta(S, K, T/DayinYr, r, PricingVol, 250, d, flag)
Moneyness = round((S-K)/K * 100) if flag == 'C' else round((K-S)/S * 100)

# st.write(f'DW Premium: {round(Premium, 2)}')
# st.write(f'TTM (Days): {T}')
# st.write(f'Moneyness (%): {Moneyness}')
# st.write(f'Sensitivity: {DWSen}')
# st.write(f'Effective Gearing: {DWGear}')
# st.write(f'Delta: {round(DWDelta, 4)}')
# st.write(f'Theta (%): {round(DWTheta / Premium * conv, 4)}')
# st.write(f'Theta (Baht): {round(DWTheta, 4)}')

print(f'DW Premium: {round(Premium, 2)}')
print(f'TTM (Days): {T}')
print(f'Moneyness (%): {Moneyness}')
print(f'Sensitivity: {DWSen}')
print(f'Effective Gearing: {DWGear}')
print(f'Delta: {round(DWDelta, 4)}')
print(f'Theta (%): {round(DWTheta / Premium * conv, 4)}')
print(f'Theta (Baht): {round(DWTheta, 4)}')

fig = getSenProfile(S, K, T/DayinYr, r, PricingVol, conv, tick, d, flag, PriceRange=99, TimeSteps=1.)
# st.plotly_chart(fig, use_container_width=True)
fig.show()
