import dash_bootstrap_components as dbc
import dash_html_components as html

def Navbar():
    navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Resume", href="/apps/resume",style={'color':'#293556'})),
     
    ],
    brand="Interactive Data Visualization",
    brand_href="/homepage",
    color="#F0FFFF",
    sticky="top",
    brand_style={'color':'#3e54d3'})
    return navbar

# def Navbar():
#     navbar=dbc.Navbar(
        
#             dbc.Row([
#                 dbc.Col(html.Img(src="/assets/python.png", height="30px")),
#                 dbc.Col(dbc.NavbarBrand("Navbar", className="ml-2",href="/homepage")),
#                 dbc.Col([dbc.NavItem(dbc.NavLink("Resume", href="/apps/resume"))],style={'margin-left':'50%'})


#             ],align="center"),
            
#         )
#     return navbar