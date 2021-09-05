import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from navbar import Navbar
from app import app
from dash_extensions import Lottie
nav = Navbar()
options = dict(loop=True, autoplay=True, rendererSettings=dict(preserveAspectRatio='xMidYMid slice'))

under_construction=Lottie(options=options, width="53%", height="50%", url='https://assets8.lottiefiles.com/packages/lf20_ojdzqkfq.json')



card_one=dbc.Card([dbc.CardBody(
       [   dbc.Row([
           dbc.Col([
               html.P("Skills demonstrated:", className="card-subtitle",style={'font-weight':'normal','color':'#c56c86'}),
               html.P(["Analysis: ","Bivariate",",","3D",html.Br(),"Hierarchical"],className="card-text"),
               dbc.CardLink("Go", href="/apps/cardOne")
           ]),
           dbc.Col([
               Lottie(options=options, width="75%", height="50%", url='https://assets9.lottiefiles.com/packages/lf20_WN7w2Z.json')

           ])
       ])
            
            
        ]
 )],color="info",outline=True,style={"width": "27rem"})
card_two=dbc.Card([dbc.CardBody(
       [
        dbc.Row([
           dbc.Col([
               html.P("Skills demonstrated:", className="card-subtitle",style={'font-weight':'normal','color':'#c56c86'}),
               html.P(["Connected Figure"],className="card-text"),
               dbc.CardLink("Go", href="/apps/cardTwo")
           ]),
           dbc.Col([
               Lottie(options=options, width="50%", height="25%", url='https://assets3.lottiefiles.com/packages/lf20_l9zJgK.json')

           ])
       ])
            
        ]
 )],color="info",outline=True,style={"width": "27rem"})
# card_three=dbc.Card([dbc.CardBody(
#        [
#             html.H4("Title", className="card-title"),
#             html.H6("tryiund", className="card-subtitle"),
#             html.P(
#                 "Some quick example text to build on the card title and make "
#                 "up the bulk of the card's content.",
#                 className="card-text",
#             ),
#             dbc.CardLink("Go", href="/apps/cardThree"),
#         ]
#  )],style={"width": "22rem"})
card_four=dbc.Card([dbc.CardBody(
       [
            dbc.Row([
           dbc.Col([
               html.P("Skills demonstrated:", className="card-subtitle",style={'font-weight':'normal','color':'#c56c86'}),
               html.P(["Animated map"],className="card-text"),
               dbc.CardLink("Go", href="/apps/cardFour")
           ]),
           dbc.Col([
               Lottie(options=options, width="75%", height="50%", url='https://assets3.lottiefiles.com/packages/lf20_kjnwk4pv.json')

           ])
       ])
            
        ]
 )],color="info",outline=True,style={"width": "25rem"})  


# card_five=dbc.Card([
#     dbc.CardHeader("DonorsChoose.org Application Screening",style={'color':'#eb6b56'}),
#        dbc.CardBody(
#        [    
#            dbc.Row([
#             dbc.Col([
#             html.H6("Skills demonstrated:", className="card-subtitle",style={'font-weight':'normal','color':'#c56c86'}),
#             html.P("NLP",className="card-text", ),
#             dbc.CardLink("Go", href="/apps/cardFive")
#             ]),
#             dbc.Col([
#                Lottie(options=options, width="50%", height="50%", url='https://assets8.lottiefiles.com/packages/lf20_0isufwmo.json')])

#            ])
#        ])


#  ],color='info',outline=True,style={"width": "27rem"})  
# card_six=dbc.Card([
#     dbc.CardHeader("Self Driving Car",style={'color':'#eb6b56'}),
#     dbc.CardBody(
#        [   dbc.Row([
#            dbc.Col([
#             html.H6("Skills demonstrated:", className="card-subtitle",style={'font-weight':'normal','color':'#c56c86'}),
#             html.P("OpenCV",className="card-text",),
#             dbc.CardLink("Go", href="/apps/cardSix"),

#            ]),
#            dbc.Col([
#                Lottie(options=options, width="50%", height="50%", url='https://assets3.lottiefiles.com/packages/lf20_kqfglvmb.json')])

#            ])
#        ])
            
# ],color='info',outline=True,style={"width": "27rem"})  
# card_seven=dbc.Card([
#     dbc.CardHeader("Facebook Friend Recommendation",style={'color':'#eb6b56'}),
#     dbc.CardBody(
#        [   dbc.Row([
#            dbc.Col([
#             html.H6("Skills demonstrated:", className="card-subtitle",style={'font-weight':'normal','color':'#c56c86'}),
#             html.P("Graph Mining",className="card-text",),
#             dbc.CardLink("Go", href="/apps/cardSeven"),

#            ]),
#            dbc.Col([
#                Lottie(options=options, width="50%", height="50%", url='https://assets4.lottiefiles.com/packages/lf20_GKNCgN.json')])

#            ])
#        ])
            
# ],color='info',outline=True,style={"width": "27rem"})   
# card_eight=dbc.Card(dbc.CardBody(
#        [
#            under_construction
            
#         ]
#  ))  

body_one=dbc.Container([ 
    dbc.Row([
        dbc.Col(card_one),
        dbc.Col(card_two),
        # dbc.Col(card_three),
        dbc.Col(card_four),
            ],justify='around',style={'margin-top':'2%'})

 ],fluid=True)

# body_two=dbc.Container([ 
#     dbc.Row([
#         dbc.Col(card_five),
#         dbc.Col(card_six),
#         dbc.Col(card_seven),
#         # dbc.Col(card_eight),
#             ],justify='around',style={'margin-top':'2%'})

#  ],fluid=True)

style={'font-weight':'normal','margin-left':'1%','margin-top':'1%'}
def Homepage():
    layout = html.Div([
    nav,
    html.H6("Interactive Data Visualization",style={'font-weight':'normal','margin-left':'1%','margin-top':'1%'}),
    body_one,
    # html.H6("Machine learning/Deep learning Projects",style={'font-weight':'normal','margin-left':'1%','margin-top':'1%'}),
    # body_two
    ])
    return layout

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )

app.layout=Homepage()

if __name__ == "__main__":
    app.run_server()