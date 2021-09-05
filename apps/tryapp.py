# import dash
# import plotly as px
# import dash_core_components as dcc
# import dash_html_components as html
# import dash_bootstrap_components as dbc
# from dash.dependencies import Output, Input
# import pandas as pd
# import plotly.graph_objects as go
# from navbar import Navbar
# from app import app
# nav = Navbar()
# data=pd.read_csv('datasets/Fortune_1000.csv')
# dff_two=data[['rank','rank_change','revenue','profit','num. of employees','prev_rank','Market Cap']]
# features_two=[]
# for i in dff_two.columns:
#     features_two.append({'label':str(i),'value':i})

# xaxisIII_dropdown=dcc.Dropdown(id='xaxisIII',options=features_two,value='rank',multi=False,style={'width':'80%'})
# yaxisIII_dropdown=dcc.Dropdown(id='yaxisIII',options=features_two,value='revenue',multi=False,style={'width':'80%'})
# zaxisIII_dropdown=dcc.Dropdown(id='zaxisIII',options=features_two,value='profit',multi=False,style={'width':'80%'})
# graphIII=dcc.Graph(id='graphplotIII',figure={})

# body_three=dbc.Container([
#     dbc.Row([
#         dbc.Col([graphIII],width={'size':9}),
#         dbc.Col([dbc.Row([xaxisIII_dropdown]),
#                 dbc.Row([yaxisIII_dropdown],style={'margin-top':'1%'}),
#                 dbc.Row([zaxisIII_dropdown],style={'margin-top':'1%'})
#                 ],align='center')
#             ])
#   ],fluid=True)

# output = html.Div(id='output',
#                   children=[],
#                   )

# def App2():
#     layout = html.Div([
#         nav,
#         # body_one,
#         # body_two,
#         body_three,
#         output,
#     ])
#     return layout
# @app.callback( 
#     Output(component_id='graphplotIII',component_property='figure'),
#     [Input(component_id='xaxisIII',component_property='value'),
#     Input(component_id='yaxisIII',component_property='value'),
#     Input(component_id='zaxisIII',component_property='value') ])
# def graph3(p,q,r):
#     data=go.Scatter3d(x=dff_two[p],y=dff_two[q],z=dff_two[r],mode='markers')
#     layout=go.Layout(title='{} vs {} vs {}'.format(p,q,r),template='plotly_white')
#     fig=go.Figure(data=data,layout=layout)
#     return fig 