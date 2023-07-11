import dash
from dash import html, dcc
import pathlib
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

app = dash.Dash()
server = app.server

import tensorflow as tf
print(tf.__version__)
PATH = pathlib.Path(__file__).parent
df_btc = pd.read_csv(PATH.joinpath("./Dataset/btc.csv").resolve())
df_eth = pd.read_csv(PATH.joinpath("./Dataset/eth.csv").resolve())
df_ada = pd.read_csv(PATH.joinpath("./Dataset/ada.csv").resolve())

df_btc["Date"]=pd.to_datetime(df_btc.Date,format="mixed")
df_eth["Date"]=pd.to_datetime(df_eth.Date,format="mixed")
df_ada["Date"]=pd.to_datetime(df_ada.Date,format="mixed")

df_btc.index=df_btc['Date']
df_eth.index=df_eth['Date']
df_ada.index=df_ada['Date']

data_btc=df_btc.sort_index(ascending=True,axis=0)
new_data_btc=pd.DataFrame(index=range(0,len(df_btc)),columns=['Date','Close'])
for i in range(0,len(data_btc)):
    new_data_btc["Date"][i]=data_btc['Date'][i]
    new_data_btc["Close"][i]=data_btc["Close"][i]

data_eth=df_eth.sort_index(ascending=True,axis=0)
new_data_eth=pd.DataFrame(index=range(0,len(df_eth)),columns=['Date','Close'])
for i in range(0,len(data_eth)):
    new_data_eth["Date"][i]=data_eth['Date'][i]
    new_data_eth["Close"][i]=data_eth["Close"][i]

data_ada=df_ada.sort_index(ascending=True,axis=0)
new_data_ada=pd.DataFrame(index=range(0,len(df_ada)),columns=['Date','Close'])
for i in range(0,len(data_ada)):
    new_data_ada["Date"][i]=data_ada['Date'][i]
    new_data_ada["Close"][i]=data_ada["Close"][i]

new_data_btc.index=new_data_btc.Date
new_data_btc.drop("Date",axis=1,inplace=True)
dataset_btc=new_data_btc.values

new_data_eth.index=new_data_eth.Date
new_data_eth.drop("Date",axis=1,inplace=True)
dataset_eth=new_data_eth.values

new_data_ada.index=new_data_ada.Date
new_data_ada.drop("Date",axis=1,inplace=True)
dataset_ada=new_data_ada.values

n_btc = int(dataset_btc.shape[0]/3) * 2
n_eth = int(dataset_eth.shape[0]/3) * 2
n_ada = int(dataset_ada.shape[0]/3) * 2

train_btc=dataset_btc[0:n_btc,:]
valid_btc=dataset_btc[n_btc:,:]

train_eth=dataset_eth[0:n_eth,:]
valid_eth=dataset_eth[n_eth:,:]

train_ada=dataset_ada[0:n_ada,:]
valid_ada=dataset_ada[n_ada:,:]

scaler1=MinMaxScaler(feature_range=(0,1))
scaler2=MinMaxScaler(feature_range=(0,1))
scaler3=MinMaxScaler(feature_range=(0,1))

scaled_data_btc=scaler1.fit_transform(dataset_btc)
scaled_data_eth=scaler2.fit_transform(dataset_eth)
scaled_data_ada=scaler3.fit_transform(dataset_ada)

x_train_btc,y_train_btc=[],[]
x_train_eth,y_train_eth=[],[]
x_train_ada,y_train_ada=[],[]

for i in range(60,len(train_btc)):
    x_train_btc.append(scaled_data_btc[i-60:i,0])
    y_train_btc.append(scaled_data_btc[i,0])
for i in range(60,len(train_eth)):
    x_train_eth.append(scaled_data_eth[i-60:i,0])
    y_train_eth.append(scaled_data_eth[i,0])
for i in range(60,len(train_ada)):
    x_train_ada.append(scaled_data_ada[i-60:i,0])
    y_train_ada.append(scaled_data_ada[i,0])
    
x_train_btc,y_train_btc=np.array(x_train_btc),np.array(y_train_btc)
x_train_btc=np.reshape(x_train_btc,(x_train_btc.shape[0],x_train_btc.shape[1],1))

x_train_eth,y_train_eth=np.array(x_train_eth),np.array(y_train_eth)
x_train_eth=np.reshape(x_train_eth,(x_train_eth.shape[0],x_train_eth.shape[1],1))

x_train_ada,y_train_ada=np.array(x_train_ada),np.array(y_train_ada)
x_train_ada=np.reshape(x_train_ada,(x_train_ada.shape[0],x_train_ada.shape[1],1))

# model_btc=load_model(PATH.joinpath("./Model/btc_model.keras").resolve())
# model_eth=load_model(PATH.joinpath("./Model/eth_model.keras").resolve())
# model_ada=load_model(PATH.joinpath("./Model/ada_model.keras").resolve())

inputs_btc=new_data_btc[len(new_data_btc)-len(valid_btc)-60:].values
inputs_btc=inputs_btc.reshape(-1,1)
inputs_btc=scaler1.transform(inputs_btc)

inputs_eth=new_data_eth[len(new_data_eth)-len(valid_eth)-60:].values
inputs_eth=inputs_eth.reshape(-1,1)
inputs_eth=scaler2.transform(inputs_eth)

inputs_ada=new_data_ada[len(new_data_ada)-len(valid_ada)-60:].values
inputs_ada=inputs_ada.reshape(-1,1)
inputs_ada=scaler3.transform(inputs_ada)

X_test_btc=[]

for i in range(60,inputs_btc.shape[0]):
    X_test_btc.append(inputs_btc[i-60:i,0])
X_test_btc=np.array(X_test_btc)


X_test_btc=np.reshape(X_test_btc,(X_test_btc.shape[0],X_test_btc.shape[1],1))
X_test_eth=[]
for i in range(60,inputs_eth.shape[0]):
    X_test_eth.append(inputs_eth[i-60:i,0])
X_test_eth=np.array(X_test_eth)
X_test_eth=np.reshape(X_test_eth,(X_test_eth.shape[0],X_test_eth.shape[1],1))

X_test_ada=[]
for i in range(60,inputs_ada.shape[0]):
    X_test_ada.append(inputs_ada[i-60:i,0])
X_test_ada=np.array(X_test_ada)
X_test_ada=np.reshape(X_test_ada,(X_test_ada.shape[0],X_test_ada.shape[1],1))

# closing_price_btc=model_btc.predict(X_test_btc)
# closing_price_btc=scaler1.inverse_transform(closing_price_btc)

# closing_price_eth=model_eth.predict(X_test_eth)
# closing_price_eth=scaler2.inverse_transform(closing_price_eth)

# closing_price_ada=model_ada.predict(X_test_ada)
# closing_price_ada=scaler3.inverse_transform(closing_price_ada)

train_btc=new_data_btc[:n_btc]
valid_btc=new_data_btc[n_btc:]
# valid_btc['Predictions']=closing_price_btc

train_eth=new_data_eth[:n_eth]
valid_eth=new_data_eth[n_eth:]
# valid_eth['Predictions']=closing_price_eth

train_ada=new_data_ada[:n_ada]
valid_ada=new_data_ada[n_ada:]
# valid_ada['Predictions']=closing_price_ada

# valid_btc.to_csv(PATH.joinpath("./Prediction/valid_btc.csv").resolve())
# valid_eth.to_csv(PATH.joinpath("./Prediction/valid_eth.csv").resolve())
# valid_ada.to_csv(PATH.joinpath("./Prediction/valid_ada.csv").resolve())

df_btc= pd.read_csv(PATH.joinpath("./Dataset/btc.csv").resolve())
df_eth= pd.read_csv(PATH.joinpath("./Dataset/eth.csv").resolve())
df_ada= pd.read_csv(PATH.joinpath("./Dataset/ada.csv").resolve())

valid_btc = pd.read_csv(PATH.joinpath("./Prediction/valid_btc.csv").resolve())
valid_eth = pd.read_csv(PATH.joinpath("./Prediction/valid_eth.csv").resolve())
valid_ada = pd.read_csv(PATH.joinpath("./Prediction/valid_ada.csv").resolve())

df_btc["Date"]=pd.to_datetime(df_btc.Date,format="mixed")
df_eth["Date"]=pd.to_datetime(df_eth.Date,format="mixed")
df_ada["Date"]=pd.to_datetime(df_ada.Date,format="mixed")

df_btc.sort_values(by=['Date'])
df_eth.sort_values(by=['Date'])
df_ada.sort_values(by=['Date'])

app.layout = html.Div([
   
    html.H1("Coin Analysis Dashboard", style={"textAlign": "center"}),
   
    dcc.Tabs(id="tabs", children=[
       
        dcc.Tab(label='Coin Prediction Data',children=[
            html.Div([
                html.H2("Actual closing price BTC",style={"textAlign": "center"}),
                dcc.Graph(
                    id="Actual Data BTC",
                    figure={
                        "data":[
                            go.Scatter(
                                x=train_btc.index,
                                y=train_btc["Close"],
                                mode='markers'
                            )
                        ],
                        "layout":go.Layout(
                            title='scatter plot',
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate'}
                        )
                    }
                ),
                html.H2("LSTM Predicted closing price BTC",style={"textAlign": "center"}),
                dcc.Graph(
                    id="Predicted Data BTC",
                    figure={
                        "data":[
                            go.Scatter(
                                x=valid_btc.index,
                                y=valid_btc["Predictions"],
                                mode='markers'
                            )
                        ],
                        "layout":go.Layout(
                            title='scatter plot',
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate'}
                        )
                    }
                )
            ]),
            html.Div([
                html.H2("Actual closing price ETH",style={"textAlign": "center"}),
                dcc.Graph(
                    id="Actual Data ETH",
                    figure={
                        "data":[
                            go.Scatter(
                                x=train_eth.index,
                                y=train_eth["Close"],
                                mode='markers'
                            )
                        ],
                        "layout":go.Layout(
                            title='scatter plot',
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate'}
                        )
                    }
                ),
                html.H2("LSTM Predicted closing price ETH",style={"textAlign": "center"}),
                dcc.Graph(
                    id="Predicted Data ETH",
                    figure={
                        "data":[
                            go.Scatter(
                                x=valid_eth.index,
                                y=valid_eth["Predictions"],
                                mode='markers'
                            )
                        ],
                        "layout":go.Layout(
                            title='scatter plot',
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate'}
                        )
                    }
                )
            ]),
            html.Div([
                html.H2("Actual closing price ADA",style={"textAlign": "center"}),
                dcc.Graph(
                    id="Actual Data ADA",
                    figure={
                        "data":[
                            go.Scatter(
                                x=train_ada.index,
                                y=train_ada["Close"],
                                mode='markers'
                            )
                        ],
                        "layout":go.Layout(
                            title='scatter plot',
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate'}
                        )
                    }
                ),
                html.H2("LSTM Predicted closing price ADA",style={"textAlign": "center"}),
                dcc.Graph(
                    id="Predicted Data ADA",
                    figure={
                        "data":[
                            go.Scatter(
                                x=valid_ada.index,
                                y=valid_ada["Predictions"],
                                mode='markers'
                            )
                        ],
                        "layout":go.Layout(
                            title='scatter plot',
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate'}
                        )
                    }
                )
            ]),
        ]),
        dcc.Tab(label='Coin Value Data', children=[
            html.Div([
                html.H1("Coin Value High vs Lows", 
                        style={'textAlign': 'center'}),
              
                dcc.Dropdown(id='my-dropdown',
                             options=[{'label': 'Bitcoin', 'value': 'BTC'},
                                      {'label': 'Ethereum','value': 'ETH'}, 
                                      {'label': 'Cardano', 'value': 'ADA'}],
                             multi=True,value=['BTC'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='highlow'),
                html.H1("Coin Volume", style={'textAlign': 'center'}),
         
                dcc.Dropdown(id='my-dropdown2',
                             options=[{'label': 'Bitcoin', 'value': 'BTC'},
                                      {'label': 'Ethereum','value': 'ETH'}, 
                                      {'label': 'Cardano', 'value': 'ADA'}],
                             multi=True,value=['BTC'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='volume')
            ], className="container"),
        ])
    ])
])

@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"BTC": "Bitcoin","ETH": "Ethereum","ADA": "Cardano",}
    trace1 = []
    trace2 = []
    for coin in selected_dropdown:
        if coin == "BTC":
            trace1.append(
              go.Scatter(x=df_btc["Date"],
                         y=df_btc["High"],
                         mode='lines', opacity=1, 
                         name=f'High {dropdown[coin]}',textposition='bottom center'))
            trace2.append(
              go.Scatter(x=df_btc["Date"],
                         y=df_btc["Low"],
                         mode='lines', opacity=1,
                         name=f'Low {dropdown[coin]}',textposition='bottom center'))
        elif coin == "ETH":
            trace1.append(
              go.Scatter(x=df_eth["Date"],
                         y=df_eth["High"],
                         mode='lines', opacity=1, 
                         name=f'High {dropdown[coin]}',textposition='bottom center'))
            trace2.append(
              go.Scatter(x=df_eth["Date"],
                         y=df_eth["Low"],
                         mode='lines', opacity=1,
                         name=f'Low {dropdown[coin]}',textposition='bottom center'))
        else:
            trace1.append(
              go.Scatter(x=df_ada["Date"],
                         y=df_ada["High"],
                         mode='lines', opacity=0.7, 
                         name=f'High {dropdown[coin]}',textposition='bottom center'))
            trace2.append(
              go.Scatter(x=df_ada["Date"],
                         y=df_ada["Low"],
                         mode='lines', opacity=0.6,
                         name=f'Low {dropdown[coin]}',textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Price (USD)"})}
    return figure
@app.callback(Output('volume', 'figure'),
              [Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown_value):
    dropdown = {"BTC": "Bitcoin","ETH": "Ethereum","ADA": "Cardano",}
    trace1 = []
    for coin in selected_dropdown_value:
        if coin == "BTC":
            trace1.append(
              go.Scatter(x=df_btc["Date"],
                         y=df_btc["Volume USDT"],
                         mode='lines', opacity=0.7,
                         name=f'Volume {dropdown[coin]}', textposition='bottom center'))
        elif coin == "ETH":
            trace1.append(
              go.Scatter(x=df_eth["Date"],
                         y=df_eth["Volume USDT"],
                         mode='lines', opacity=0.7,
                         name=f'Volume {dropdown[coin]}', textposition='bottom center'))
        else:
            trace1.append(
              go.Scatter(x=df_ada["Date"],
                         y=df_ada["Volume USDT"],
                         mode='lines', opacity=0.7,
                         name=f'Volume {dropdown[coin]}', textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data, 
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M',
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Transactions Volume"})}
    return figure

if __name__ == '__main__':
    app.run_server(debug=True)
