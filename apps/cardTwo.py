import dash
import plotly.express as px
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input
import pandas as pd
import plotly.graph_objects as go
from navbar import Navbar
from app import app
nav = Navbar()
data=pd.read_csv('datasets/cpse.csv')
df = px.data.gapminder()
Intro =dbc.Card(dbc.CardBody(
       [
        
            html.P(
                "Select the value(s) from dropdown , hover over the markers to observe percentage population distribution on that particular year"
            )
            
        ]
 ),style={'margin-top':'1%','margin-left':'1%','margin-right':'1%'})

dropdown_menu=dcc.Dropdown(id='dpdwn',value=['India','China'], multi=True,
                 options=[{'label': x, 'value': x} for x in
                          df.country.unique()])
donut_graph=dcc.Graph(id='donut',figure={})
line_graph=dcc.Graph(id='line',figure={})

body_one=dbc.Container([
    dbc.Row([
        dbc.Col([
            dropdown_menu
         ],width={'size':12})
     ],style={'margin-top':'1%'}),
    dbc.Row([
        dbc.Col([line_graph],width=8),
        dbc.Col([donut_graph]) 
        ]),
    
     ],fluid=True,style={'margin-top':'1%'})

output = html.Div(id='output',
                  children=[],
                  )
def App_two():
    layout = html.Div([
        nav,
        Intro,
        body_one,
        output,
    ])
    return layout

@app.callback(
    Output(component_id='line', component_property='figure'),
    [Input(component_id='dpdwn', component_property='value')])
def update_line_graph(country):
    dff = df[df.country.isin(country)]
    fig = px.line(data_frame=dff, x='year', y='gdpPercap', color='country',
                  custom_data=['country', 'continent', 'lifeExp', 'pop'],template='simple_white')
    fig.update_traces(mode='lines+markers',opacity=0.65)
    
    return fig 

@app.callback(
    Output(component_id='donut', component_property='figure'),
    Input(component_id='line', component_property='hoverData'),
    Input(component_id='line', component_property='clickData'),
    Input(component_id='line', component_property='selectedData'),
    Input(component_id='dpdwn', component_property='value')
) 

def update_donut_graph(hov_data, clk_data, slct_data, country_chosen):
    if hov_data is None:
            dff2 = df[df.country.isin(country_chosen)]
            dff2 = dff2[dff2.year == 1952]
          
            fig2 = px.pie(data_frame=dff2, values='pop', names='country',
                      title='Population for 1952',hole=.3,color_discrete_sequence=px.colors.sequential.Darkmint)
            return fig2
    else:
        
        dff2 = df[df.country.isin(country_chosen)]
        hov_year = hov_data['points'][0]['x']
        dff2 = dff2[dff2.year == hov_year]
        fig2 = px.pie(data_frame=dff2, values='pop', names='country', title=f'Population for: {hov_year}',hole=.3,color_discrete_sequence=px.colors.sequential.Darkmint)
        return fig2
