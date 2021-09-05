# import dash
# import plotly.express as px
# import dash_core_components as dcc
# import dash_html_components as html
# import dash_bootstrap_components as dbc
# from dash.exceptions import PreventUpdate
# from dash.dependencies import Output, Input
# import pandas as pd
# from dash_extensions import Lottie
# # import matplotlib.pyplot as plt
# import numpy as np
# import pickle
# import warnings
# warnings.filterwarnings("ignore")
# import plotly.graph_objects as go
# # from wordcloud import WordCloud, STOPWORDS
# from navbar import Navbar
# from app import app
# nav = Navbar()
# options = dict(loop=True, autoplay=True, rendererSettings=dict(preserveAspectRatio='xMidYMid slice'))
# # X=html.Video(src='/assets/steering.m4v',id='v2',children='steering')
# # def change_one(x):
# #     frame_one=html.Video(src='/assets/car.m4v',autoPlay=x,id='v1')
# #     return frame_one
# # def change_two(x):
# #     frame_two=html.Video(src='/assets/steering.m4v',autoPlay=x,id='v2')
# #     return frame_two


# body=dbc.Container([
#     dbc.Row([
#         dbc.Col([
#             dbc.Card([dbc.CardHeader("about"), 
#             dbc.CardBody(html.P([" In case file is not playing",html.Br(),'Click on the Homepage(Interactive Data Visualization and ML/DL Projects),then again click on the link   ',  html.Code('Go'), '  of Self Driving Car']))

#             ])
#         ],width=7),
#         dbc.Col([
#             dbc.Card([dbc.CardHeader("output"),
#                 dbc.CardBody([html.Video(src='/assets/Drive_COMP.mp4',controls=True,height='100%',width='100%')])
                
#                 # dbc.Button(children="CLICK HERE",id="btn",n_clicks=0)
#             ])

# ],width=5)
#     ],style={'margin-top':'1%'})
# ],fluid=True)

# output = html.Div(id='output',
#                   children=[],
#                   )

# def App_six():
#     layout = html.Div([
#         nav,
#         body
#     ])
#     return layout


# # @app.callback([Output('v1','children'),Output('v2','children')],
# # [Input('btn','n_clicks')])

# # def Play_video(n):
# #     if n==1:
# #         App_six()
# #         change_one(True)
# #         change_two(True)
# #     raise PreventUpdate
        