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

graphIcolor='#FD836D'
data=pd.read_csv('datasets/cpse.csv')
Intro =dbc.Card(dbc.CardBody(
       [
            html.P(["Figure (i): Select the values from dropdowns to determine the empirical relationship between them.",html.Br(),"Figure (ii): Click on the concentric circles,Sunburst plots visualize hierarchical data spanning outwards radially from root to leaves.",html.Br(),"Figure (iii): Select the values from dropdowns to determine the empirical relationship among them."],
                className="card-text"
            ),
            
        ]
 ))

dff=data
features=[]
for i in dff.columns:
    features.append({'label':str(i),'value':i})
dff_sun=data[['ONGC', 'OIL', 'GAIL', 'IOCL',
       'BPCL', 'HPCL', 'CPCL', 'MRPL', 'NRL', 'BALMER LAWRIE', 'BIECCO LAWRIE',
       'EIL', 'TOTAL (P)']]
sunburn=[]
for j in dff_sun.columns:
    sunburn.append({'label':str(j),'value':j})


xaxis_dropdown=dcc.Dropdown(id='xaxis',options=features,value='OIL',multi=False,placeholder='X axis',style={'width':'80%'})
yaxis_dropdown=dcc.Dropdown(id='yaxis',options=features,value='NRL',multi=False,placeholder='Y axis',style={'width':'80%'})
graphI=dcc.Graph(id='graphPlotI',figure={})
xaxisII_dropdown=dcc.Dropdown(id='xaxisII',options=sunburn,value='ONGC',multi=False,style={'width':'100%'})
graphII=dcc.Graph(id='graphplotII',figure={})
xaxisIII_dropdown=dcc.Dropdown(id='xaxisIII',options=features,value='IOCL',multi=False,style={'width':'80%'})
yaxisIII_dropdown=dcc.Dropdown(id='yaxisIII',options=features,value='HPCL',multi=False,style={'width':'80%'})
zaxisIII_dropdown=dcc.Dropdown(id='zaxisIII',options=features,value='CPCL',multi=False,style={'width':'80%'})
graphIII=dcc.Graph(id='graphplotIII',figure={})

body_one=dbc.Container([
    dbc.Row([ 
        dbc.Col(Intro)
    ],style={'margin-top':'1%'}),
    dbc.Row([
         dbc.Col([graphI],width={'size':9}),  
         dbc.Col([dbc.Row([xaxis_dropdown]),
                dbc.Row([yaxis_dropdown],style={'margin-top':'1%'})],align='center')
     ],style={'margin-top':'1%'}) 
 ],fluid=True)

body_two=dbc.Container([
    dbc.Row([
        dbc.Col([xaxisII_dropdown],align='center'),
        dbc.Col([graphII],width={'size':9})
     ])

 ])
body_three=dbc.Container([
    dbc.Row([
        dbc.Col([graphIII],width={'size':9}),
        dbc.Col([dbc.Row([xaxisIII_dropdown]),
                dbc.Row([yaxisIII_dropdown],style={'margin-top':'1%'}),
                dbc.Row([zaxisIII_dropdown],style={'margin-top':'1%'})
                ],align='center')
            ])
  ],fluid=True)

output = html.Div(id='output',
                  children=[],
                  )

def App_one():
    layout = html.Div([
        nav,
        body_one,
        body_two,
        body_three,
        output,
    ])
    return layout
#first  
@app.callback( 
    Output(component_id='graphPlotI',component_property='figure'),
    [Input(component_id='xaxis',component_property='value'),
    Input(component_id='yaxis',component_property='value') ])
def graphI(xaxis,yaxis):
    data=go.Scatter(x=dff[xaxis],y=dff[yaxis],mode='markers',marker={'size':20,'color':px.colors.qualitative.Bold})
    layout=go.Layout(title='Figure (i) {} vs {}'.format(xaxis,yaxis),xaxis={'title':xaxis},yaxis={'title':yaxis},template='simple_white')
    fig=go.Figure(data=data,layout=layout)
    return fig  
#second
@app.callback( 
    Output(component_id='graphplotII',component_property='figure'),
    [Input(component_id='xaxisII',component_property='value')])
def graph2(xaxisII):
    y=dff[xaxisII]
    fig=px.sunburst(
    data_frame=dff,
    path=['CATEGORY','SECTOR_CATEGORY_CPSE',y],color='SECTOR_CATEGORY_CPSE',
    color_discrete_sequence=px.colors.qualitative.Bold,branchvalues="total",
    title="Figure (ii) Distribution of CPSE : {}".format(xaxisII),height=650)
    fig.update_traces(textinfo='label+percent entry')
    fig.update_traces(insidetextorientation='tangential')
    return fig
#third
@app.callback( 
    Output(component_id='graphplotIII',component_property='figure'),
    [Input(component_id='xaxisIII',component_property='value'),
    Input(component_id='yaxisIII',component_property='value'),
    Input(component_id='zaxisIII',component_property='value') ])
def graph3(p,q,r):
    data=go.Scatter3d(x=dff[p],y=dff[q],z=dff[r],mode='markers')
    layout=go.Layout(title='Figure (iii) {} vs {} vs {}'.format(p,q,r),template='plotly_white',height=650)
    fig=go.Figure(data=data,layout=layout)
    return fig 
    