import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# import plotly.io as pio
# pio.renderers.default='browser' # browser, svg


## Import Data ##
# filename = 'M_VAlue 1.xlsx'
filename = 'M_VAlue 1.xlsx'
data = pd.read_excel(filename)

st.set_page_config(page_title='DW Dashboard',
                   layout='wide')

st.sidebar.header('Input')

## Create DataFrame ##
df = data.copy()
df.columns = ['Date', 'Stock', 'Volume', 'Value']
df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
df = df.set_index('Date')

with st.sidebar:
    window = st.number_input("Smoothing (Day)", value=21, min_value=1, max_value=200, placeholder="Type a number...")

df['SumVA'] = df.groupby('Date')['Value'].transform(lambda x: x.sum())
df['PctVA'] = df.Value / df.SumVA
df['RS'] = df.groupby('Stock')['PctVA'].transform(lambda x: (x - x.rolling(window).mean()) / x.rolling(window).std()) + 100
df['RawMoM'] = df.groupby('Stock')['RS'].transform(lambda x: (x.pct_change()))
df['MoM'] = df.groupby('Stock')['RawMoM'].transform(lambda x: (x - x.rolling(window).mean()) / x.rolling(window).std()) + 100

## Plot RRG ##
AllDate = np.unique(df.index)

with st.sidebar:
    TargetLookBack = st.selectbox("RRG Target Date", options=sorted(AllDate, reverse=True))
    TailLookBack = st.number_input("RRG Tails (Day)", value=1, min_value=1, max_value=5, placeholder="Type a number...")

TargetIndex = len(AllDate) - np.where(TargetLookBack == AllDate)[0][0]
TailDate = np.unique(df.index)[-(TailLookBack + TargetIndex)]
TargetDate = np.unique(df.index)[-TargetIndex]

plot_df = df[df.index >= TailDate].copy()

x_axis = 'RS'
y_axis = 'MoM'


Scatter = plot_df[plot_df.index == TargetDate]
Tail = plot_df[(plot_df.index >= TailDate) & (plot_df.index <= TargetDate)]

fig_Scatter = px.scatter(Scatter, x=x_axis, y=y_axis,
                color='Stock',
                title=f'VA RRG')

fig_Tail = px.line(Tail, x=x_axis, y=y_axis,
                color='Stock',
                title=f'VA RRG',
                line_shape='spline',
                markers=False)


fig_rrg = go.Figure(data=fig_Scatter.data + fig_Tail.data)

fig_rrg.update_traces(marker={'size': 10},
                      line={'width': 1},
                          textposition='top right')

fig_rrg.update_layout(font=dict(size=10,
                              color='white'),
                      height=700,
                    xaxis_title=f'{x_axis}',
                    yaxis_title=f'{y_axis}',
                    title=f'RRG VA As of Date: {TargetDate}'
                    )

fig_rrg.add_vline(x=100, line_color='red')
fig_rrg.add_hline(y=100, line_color='red')

fig_rrg.update_layout({
    'plot_bgcolor': 'rgb(0,0,0,0)',
    'paper_bgcolor': '#222A2A'
})
fig_rrg.update_xaxes(showgrid=True, gridwidth=0.01, gridcolor='#0D2A63')
fig_rrg.update_yaxes(showgrid=True, gridwidth=0.01, gridcolor='#0D2A63')
# fig["layout"].pop("updatemenus")

# st.plotly_chart(fig_rrg, use_container_width=True)

## Plot VA Rank ##
with st.sidebar:
    TargetLookBack = st.selectbox("Percentage VA Target Date", options=sorted(AllDate, reverse=True))
    LookBackPeriod = st.number_input("Percentage VA Lookback (Day)", value=5, min_value=1, max_value=365, placeholder="Type a number...")

TargetIndex = len(AllDate) - np.where(TargetLookBack == AllDate)[0][0]
TargetDate = np.unique(df.index)[-TargetIndex]

plot_df = df.copy()
plot_df['LookBackPctVA'] = -plot_df.PctVA.shift(LookBackPeriod)
plot_df = plot_df.dropna()
AvgPctVA = np.percentile(plot_df.PctVA, 85)
AvgLookBackPctVA = np.percentile(plot_df.LookBackPctVA, 15)

fig_VAcomp = px.bar(plot_df[plot_df.index == AllDate[-(TargetIndex)]].sort_values('PctVA', ascending=False), x='Stock', y=['PctVA', 'LookBackPctVA'],
              color_discrete_map={'PctVA': 'green',
                                  'LookBackPctVA': 'red'},
              title=f'Percentage VA As of Date: {AllDate[-TargetIndex]}')

fig_VAcomp.update_layout(xaxis_title='Stock',
                  yaxis_title='Percentage VA',
                  legend_title='Stock',
                  hovermode="x unified",
                  height=700,
                  font=dict(size=10,
                            color='white'))

fig_VAcomp.add_hline(y=AvgPctVA, line_color='white', line_dash='dash')
fig_VAcomp.add_hline(y=AvgLookBackPctVA, line_color='white', line_dash='dash')

fig_VAcomp.update_xaxes(showgrid=True, gridwidth=0.01, gridcolor='#0D2A63', ticklabelstep=1)
fig_VAcomp.update_yaxes(showgrid=True, gridwidth=0.01, gridcolor='#0D2A63')

fig_VAcomp.update_layout({
    'plot_bgcolor': 'rgb(0,0,0,0)',
    'paper_bgcolor': '#222A2A'
})

# st.plotly_chart(fig_VAcomp, use_container_width=True)

# left_column, right_column = st.columns(2)
st.plotly_chart(fig_rrg, use_container_width=True)
st.plotly_chart(fig_VAcomp, use_container_width=True)

## Plot VA Heat ##
with st.sidebar:
    window = st.number_input("Smoothing HeatBar (Day)", value=21, min_value=1, max_value=100, placeholder="Type a number...")
    LookBack = st.number_input("Lookback HeatBar (Day)", value=60, min_value=21, max_value=90, placeholder="Type a number...")

TargetDate = np.unique(df.index)[-LookBack]
plot_df = df[df.index >= TargetDate].copy()

plot_df['AvgValue'] = plot_df.groupby('Stock')['Value'].transform(lambda x: x.rolling(window).mean())
plot_df['Stdev'] = plot_df.groupby('Stock')['Value'].transform(lambda x: x.rolling(window).std())
plot_df['OneStdev'] = plot_df.AvgValue + plot_df.Stdev
plot_df['TwoStdev'] = plot_df.AvgValue + 2 * plot_df.Stdev
plot_df = plot_df.dropna()

AllStock = np.unique(plot_df.Stock)

Color = []
for x in range(len(plot_df)):
    if ((plot_df['Value'][x] >= plot_df['AvgValue'][x]) & (plot_df['Value'][x] < plot_df['OneStdev'][x])):
        Color.append('#3366CC')
    elif ((plot_df['Value'][x] >= plot_df['OneStdev'][x]) & (plot_df['Value'][x] < plot_df['TwoStdev'][x])):
        Color.append('rgb(255,217,47)')
    elif ((plot_df['Value'][x] >= plot_df['TwoStdev'][x])):
        Color.append('#EF553B')
    else: 
        Color.append('grey')

plot_df['Color'] = Color

Stock_Select = st.sidebar.multiselect(
    'Select Stock',
    options=AllStock,
    default='AAV')

plot_df_selection = plot_df.query(
    'Stock == @Stock_Select'
    )

TargetStock = np.unique(plot_df_selection.Stock)


from plotly.subplots import make_subplots

fig = make_subplots(rows=len(TargetStock), cols=1, subplot_titles=np.unique(TargetStock),) #row_heights=row_heights)

for i in range(len(TargetStock)):
    fig.add_trace(
        go.Bar(x=plot_df_selection[plot_df_selection.Stock == TargetStock[i]].index,
              y=plot_df_selection[plot_df_selection.Stock == TargetStock[i]]['Value'],
              marker_color=plot_df_selection[plot_df_selection.Stock == TargetStock[i]]['Color'],
               name='Value',
              showlegend=True),
        row=i+1,col=1
    )
    
fig.update_layout(height=500,
                  title_text='Value Analysis',
                  hovermode="x unified",
                  barmode='stack',
                  font=dict(size=12,
                              color='white'),
                  hoverlabel=dict(bgcolor="rgb(0,0,0)"))

fig.update_xaxes(showgrid=True, gridwidth=0.01, gridcolor='#0D2A63')
fig.update_yaxes(showgrid=True, gridwidth=0.01, gridcolor='#0D2A63')

fig.update_layout({
    'plot_bgcolor': 'rgb(0,0,0,0)',
    'paper_bgcolor': '#222A2A'
})


dt_all = pd.date_range(start=np.unique(df.index)[0],end=np.unique(df.index)[-1])
dt_obs = [d.strftime("%Y-%m-%d") for d in pd.to_datetime(plot_df.index)]
dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d").tolist() if not d in dt_obs]
fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])

st.plotly_chart(fig, use_container_width=True)
