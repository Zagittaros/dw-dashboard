import numpy as np
import pandas as pd
from scipy.stats import norm
import plotly.express as px
import streamlit as st

def getd(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return d1, d2

def getDelta(S, K, T, r, sigma, d=0., flag='C'):
    N = norm.cdf
    d1, d2 = getd(S, K, T, r, sigma)
    
    if flag == 'C':
        return N(d1)
    elif flag == 'P':
        return N(d1) - 1
    
def getTheta(S, K, T, r, sigma, OptionDays=250, d=0., flag='C'):
    N = norm.cdf
    d1, d2 = getd(S, K, T, r, sigma)

    Nd1 = np.exp(-(d1**2)/2) / (np.sqrt(2*np.pi))
    
    if flag == 'C':
        return 1/OptionDays * -((S*sigma*Nd1) / (2*np.sqrt(T)) - r*K*np.exp(-r*T)*N(d2))
    elif flag == 'P':
        return 1/OptionDays * -((S*sigma*Nd1) / (2*np.sqrt(T)) + r*K*np.exp(-r*T)*(1-N(d2)))
    
def BSPrice(S, K, T, r, sigma, d=0., flag='C'):
    N = norm.cdf
    d1, d2 = getd(S, K, T, r, sigma)
    
    if flag == 'C':
        return S * N(d1) - K * np.exp(-r*T) * N(d2)
    elif flag == 'P':
        return K * np.exp(-r*T) * N(-d2) - S * N(-d1)
    
def getSen(S, K, T, r, sigma, conv, tick=0.25, d=0., flag='C'):
    delta = getDelta(S, K, T, r, sigma, d, flag)
    return round((delta/conv) * (tick/0.01), 3)

def getGearing(S, K, T, r, sigma, conv, d=0., flag='C'):
    delta = getDelta(S, K, T, r, sigma, d, flag)
    Premium = BSPrice(S, K, T, r, sigma, d, flag)
    return abs(delta) * (S/Premium)

def getSenProfile(S, K, T, r, sigma, conv, tick=0.25, d=0., flag='C', PriceRange=99, TimeSteps=1.):
    DayinYr = 365
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

## Play with this part ##
st.set_page_config(page_title='DW Dashboard',
                   layout='wide')

st.sidebar.header('Input')

DayinYr = 365
OptionDays = 250

with st.sidebar:
    S = st.number_input("Spot", value=70., min_value=1., placeholder="Type a number...")
    K = st.number_input("Strike", value=75.5, min_value=1., placeholder="Type a number...")
    T = st.number_input("TTM (Days)", value=118, min_value=1, placeholder="Type a number...")
    r = st.number_input("Discount Rate", value=0.025, min_value=0., placeholder="Type a number...")
    sigma = st.number_input("Volatility", value=0.61, min_value=0., placeholder="Type a number...")
    conv = st.number_input("Conversion Rate", value=0.08333, min_value=0.0001, placeholder="Type a number...")
    tick = st.number_input("Tick Size", value=0.25, min_value=0.01, placeholder="Type a number...")
    d = st.number_input("Dividend Yield", value=0., min_value=0., placeholder="Type a number...")
    flag = st.selectbox("Call/Put", ('C', 'P'))
    PriceRange = st.number_input("Simulation Range", value=99, min_value=15, placeholder="Type a number...")
    TimeSteps = st.number_input("Simulation Step", value=1, min_value=1, placeholder="Type a number...")

fig = getSenProfile(S, K, T/DayinYr, r, sigma, 1/conv, tick, d, flag, PriceRange=99, TimeSteps=1.)
st.plotly_chart(fig, use_container_width=True)
