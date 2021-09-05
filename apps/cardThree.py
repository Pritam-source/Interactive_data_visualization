# import dash
# import plotly.express as px
# import dash_core_components as dcc
# import dash_html_components as html
# import dash_bootstrap_components as dbc
# from dash.dependencies import Output, Input
# import pandas as pd
# import plotly.graph_objects as go
# from navbar import Navbar
# from app import app
# import dash_cytoscape as cyto
# nav = Navbar()


# body_one= html.Div([
#     cyto.Cytoscape(
#         id='cytoscape-two-nodes',
#         layout={'name': 'preset'},
#         style={'width': '100%', 'height': '400px'},
#         elements=[
#             {'data': {'id': 'one', 'label': 'Node 1'}, 'position': {'x': 75, 'y': 75}},
#             {'data': {'id': 'two', 'label': 'Node 2'}, 'position': {'x': 200, 'y': 200}},
#             {'data': {'source': 'one', 'target': 'two'}}
#         ]
#     )
# ])


# body_two = html.Div([
#     html.H3("Try")
#  ],className="banner")




# output = html.Div(id='output',
#                   children=[],
#                   )
# def App_three():
#     layout = html.Div([
#         nav,
#         body_one,
#         body_two,
#         output,
#     ])
#     return layout