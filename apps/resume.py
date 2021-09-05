import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash_extensions import Lottie
from navbar import Navbar
from app import app
nav = Navbar()

options = dict(loop=True, autoplay=True, rendererSettings=dict(preserveAspectRatio='xMidYMid slice'))
data_science=html.Div([dbc.Row([ 
                    dbc.Col([
                    Lottie(options=options, width="50%", height="75%", url='https://assets5.lottiefiles.com/private_files/lf30_m1od7oor.json')],width=1),
                    dbc.Col([html.H6("Data Science")],align='center')
                              ])   
                    ])
web_dev=html.Div([dbc.Row([ 
                    dbc.Col([
                    Lottie(options=options, width="50%", height="75%", url='https://assets9.lottiefiles.com/packages/lf20_rxuub8j6.json')],width=1),
                    dbc.Col([
                        dbc.Row([html.H6("Data oriented Web Apps (Model deployment, Data visualization)")],align='center'),
                        dbc.Row([html.H6("Web Development FULL-STACK")])
                        ])
                    ])
                     ])  
                    

Intro =dbc.Card([dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Row([html.H6("Name    : Pritam")]),
                    dbc.Row([html.H6("Email   : pritamkumar747@gmail.com")]),
                    dbc.Row([html.H6("Phone  : +91 9038763933")]),
                    ]), 
                dbc.Col(dbc.Row([Lottie(options=options, width="25%", height="25%", url='https://assets10.lottiefiles.com/packages/lf20_rycdh53q.json')]))
                  ])

                              ])
                ],style={'margin-top':'1%','margin-left':'1%','margin-right':'1%'})
           
body_one=dbc.Card([dbc.CardBody([
    dbc.Row([
        dbc.Col([
            dbc.Row([html.H6("MATHEMATICAL REASONING:")]),
            dbc.Row([html.Img(src='/assets/calculus.png',style={'width':'8%'}),
                     html.H6("Calculus",style={'margin-left':'3%','margin-top':'3%'}) ]),
            dbc.Row([html.Img(src='/assets/sta&prob.png',style={'width':'8%'}),
                     html.H6("Statistics and Probability",style={'margin-left':'3%','margin-top':'3%'}) ],style={'margin-top':'1%'}),
            dbc.Row([html.Img(src='/assets/linear algebra.png',style={'width':'8%'}),
                     html.H6("Linear Algebra",style={'margin-left':'3%','margin-top':'3%'}) ],style={'margin-top':'1%'})
            
                ]),
        dbc.Col([
            dbc.Row([html.H6("PROGRAMMING:")]),
            dbc.Row([html.Img(src='/assets/python.png',style={'width':'10%'}),
                html.H6("Python",style={'margin-left':'3%','margin-top':'3%'}) ])
                ]),
        dbc.Col([
            dbc.Row([html.H6("TOOLS:")]),
            dbc.Row([html.Img(src='/assets/pandas.png',style={'width':'10%'}),
                html.H6("Pandas",style={'margin-left':'3%','margin-top':'3%'}) ]),
            dbc.Row([html.Img(src='/assets/numpy.png',style={'width':'10%'}),
                html.H6("NumPy",style={'margin-left':'3%','margin-top':'3%'}) ]),
            dbc.Row([html.Img(src='/assets/Scipy.png',style={'width':'10%'}),
                html.H6("SciPy",style={'margin-left':'3%','margin-top':'3%'}) ]),
            dbc.Row([html.Img(src='/assets/table.png',style={'width':'10%'}),
                html.H6("Tableau",style={'margin-left':'3%','margin-top':'3%'}) ])
                ]),
        dbc.Col([
            dbc.Row([html.H6("ML/DL:")]),
            dbc.Row([html.Img(src='/assets/scikit.png',style={'width':'10%'}),
                html.H6("scikit-learn",style={'margin-left':'3%','margin-top':'3%'}) ]),
            ]),
        dbc.Col([
             dbc.Row([html.H6("VISUALIZATION:")]),
            dbc.Row([html.Img(src='/assets/matplotlib.png',style={'width':'10%'}),
                html.H6("Matplotlib",style={'margin-left':'3%','margin-top':'3%'}) ]),
            dbc.Row([html.Img(src='/assets/seaborn.png',style={'width':'10%'}),
                html.H6("seaborn",style={'margin-left':'3%','margin-top':'3%'}) ]),
            dbc.Row([html.Img(src='/assets/plotly.png',style={'width':'10%'}),
                html.H6("Plotly",style={'margin-left':'3%','margin-top':'3%'}) ]),
            dbc.Row([html.Img(src='/assets/table.png',style={'width':'10%'}),
                html.H6("Tableau",style={'margin-left':'3%','margin-top':'3%'}) ])
                ]),
        dbc.Col([
            dbc.Row([html.H6("CLOUD COMPUTING:")]),
            dbc.Row([html.Img(src='/assets/azure.png',style={'width':'10%'}),
                html.H6("Azure machine learning",style={'margin-left':'3%','margin-top':'3%'}) ])

         ]),
        dbc.Col([
            dbc.Row([html.H6("BIG DATA:")]),
            dbc.Row([html.Img(src='/assets/apache.png',style={'width':'10%'}),
                html.H6("Apache Spark",style={'margin-left':'3%','margin-top':'3%'}) ])
        ])
     ]),
     
 ])
],style={'margin-left':'1%','margin-right':'1%'})

body_two=dbc.Card([dbc.CardBody([
    dbc.Row([
        dbc.Col([
            dbc.Row([html.H6("BACK-END:")]),
            dbc.Row([html.Img(src='/assets/django.png',style={'width':'11%'}),
                     html.H6("Django",style={'margin-left':'3%','margin-top':'3%'}) ]),
            dbc.Row([html.Img(src='/assets/flask.png',style={'width':'11%'}),
                     html.H6("Flask",style={'margin-left':'3%','margin-top':'3%'}) ]),
            dbc.Row([html.Img(src='/assets/heroku.png',style={'width':'11%'}),
                     html.H6("Heroku",style={'margin-left':'3%','margin-top':'3%'}) ])
         ]),
        dbc.Col([
            dbc.Row([html.H6("FRONT-END:")]),
            dbc.Row([html.Img(src='/assets/plotly.png',style={'width':'7%'}),
                     html.H6("Dash Core Components",style={'margin-left':'3%','margin-top':'3%'}) ]),
            dbc.Row([html.Img(src='/assets/html.png',style={'width':'7%'}),
                     html.H6("Dash HTML Components",style={'margin-left':'3%','margin-top':'3%'}) ]),
            dbc.Row([html.Img(src='/assets/bootstrap.png',style={'width':'7%'}),
                     html.H6("Dash Bootstrap Components",style={'margin-left':'3%','margin-top':'3%'}) ]),
            dbc.Row([html.Img(src='/assets/plotly.png',style={'width':'7%'}),
                     html.H6("Dash DataTable",style={'margin-left':'3%','margin-top':'3%'}) ]),
            dbc.Row([html.Img(src='/assets/css.png',style={'width':'7%'}),
                     html.H6("CSS",style={'margin-left':'3%','margin-top':'3%'}) ])

         ]),
        dbc.Col([
            dbc.Row([html.H6("DATABASE:")]),
            dbc.Row([html.Img(src='/assets/sqlite.png',style={'width':'10%'}),
                     html.H6("SQLite",style={'margin-left':'3%','margin-top':'3%'}) ]),
            dbc.Row([html.Img(src='/assets/postgresql.png',style={'width':'10%'}),
                     html.H6("PostgreSQL",style={'margin-left':'3%','margin-top':'3%'}) ])

         ])
     ])
 ])

 ],style={'margin-left':'1%','margin-right':'1%'})






output = html.Div(id='output',
                  children=[],
                  )

def App_resume():
    layout = html.Div([
        nav,
        Intro,
        data_science,
        body_one,
        web_dev,
        body_two,
        output,
    ])
    return layout