import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from apps.cardOne import App_one
from apps.cardTwo import App_two
# from apps.cardThree import App_three
from apps.cardFour import App_four
# from apps.cardFive import App_five
# from apps.cardSix  import App_six
# from apps.cardSeven import App_seven
from apps.resume import App_resume
from homepage import Homepage 
from app import app
from app import server
app.layout = html.Div([
    dcc.Location(id = 'url', refresh = False),
    html.Div(id = 'page-content',children=[])
])  
@app.callback(Output('page-content', 'children'),
            [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/apps/cardOne':
        return App_one()
    if pathname == '/apps/cardTwo':
        return App_two()
    # if pathname == '/apps/cardThree':
    #     return App_three()
    if pathname == '/apps/cardFour':
        return App_four()
    # if pathname == '/apps/cardFive':
    #     return App_five()
    # if pathname == '/apps/cardSix':
    #     return App_six()
    # if pathname == '/apps/cardSeven':
    #     return App_seven()
    if pathname == '/apps/resume':
        return App_resume()
    
    else:
        return Homepage()
if __name__ == '__main__':
    app.run_server(debug=False)
