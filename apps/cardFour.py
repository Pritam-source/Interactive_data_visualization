import dash
import plotly.express as px
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input
import pandas as pd
import plotly.graph_objects as go
import configparser
from navbar import Navbar
from app import app
nav = Navbar()
config = configparser.ConfigParser()
config.read('config.ini')
mapbox_token = config['mapbox']['secret_token']
px.set_mapbox_access_token(mapbox_token)
data_url = 'https://shahinrostami.com/datasets/time-series-19-covid-combined.csv'
data = pd.read_csv(data_url)
missing_states = pd.isnull(data['Province/State'])
data.loc[missing_states,'Province/State'] = data.loc[missing_states,'Country/Region']
data['Active'] = data['Confirmed'] - data['Recovered'] - data['Deaths']
data = data.dropna()
date_mask = data['Date'] == data['Date'].max()
features=data[['Active','Confirmed','Recovered','Deaths']]
feature=[]
for i in features.columns:
    feature.append({'label':str(i),'value':i})
graph=dcc.Graph(id='graphmap',figure={})
dropdn=dcc.Dropdown(id='dpdn',options=feature,multi=False,value='Active')

Intro =dbc.Card(dbc.CardBody(
       [
            html.P(["Choose the option from dropdown and click on play button to visualize the selected option cases of coronavirus ",html.Br(),"*"])
            
        ]
 ),style={'margin-top':'1%','margin-left':'1%','margin-right':'1%'})


body_one=dbc.Container([ 
    dbc.Row([
        dbc.Col([graph],width={'size':9}),
        dbc.Col([dropdn],align='center',width={'size':2})
     ])
],fluid=True)

output = html.Div(id='output',
                  children=[],
                  )
def App_four():
    layout = html.Div([
        nav,
        Intro,
        body_one,
        output,
    ])
    return layout

@app.callback(
    Output(component_id='graphmap',component_property='figure'),
    [Input(component_id='dpdn',component_property='value')]
)
def update_graph(cat):
    if cat=='Active':
        fig = px.scatter_mapbox(
        data, lat="Lat", lon="Long",
        size="Active", size_max=50,
        color="Deaths", color_continuous_scale=px.colors.sequential.Pinkyl,
        hover_name="Province/State",           
        mapbox_style='dark', zoom=1,
        animation_frame="Date", animation_group="Province/State",
        title="Active"
)
    else:
        fig = px.scatter_mapbox(
        data, lat="Lat", lon="Long",
        size=cat, size_max=50,
        color="Deaths", color_continuous_scale=px.colors.sequential.Pinkyl,
        hover_name="Province/State",           
        mapbox_style='dark', zoom=1,
        animation_frame="Date", animation_group="Province/State",
        title=cat
        )
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 200
    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 200
    fig.layout.coloraxis.showscale = False
    fig.layout.sliders[0].pad.t = 10
    fig.layout.updatemenus[0].pad.t= 10
    return fig