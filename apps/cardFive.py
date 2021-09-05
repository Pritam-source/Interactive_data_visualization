# import dash
# import plotly.express as px
# import dash_core_components as dcc
# import dash_html_components as html
# import dash_bootstrap_components as dbc
# from dash.dependencies import Output, Input
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics import confusion_matrix
# from sklearn import metrics
# from sklearn.preprocessing import Normalizer
# from sklearn.metrics import roc_curve, auc
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
# data_url='https://raw.githubusercontent.com/Pritam-source/data_source/main/preprocessed_data_web.csv'
# df = pd.read_csv(data_url)
# Intro=dbc.Container([html.P(
#                 "DonorsChoose.org receives hundreds of thousands of project proposals each year for classroom projects in need of funding. Right now, a large number of volunteers is needed to manually screen each submission before it's approved to be posted on the DonorsChoose.org website.",className="Intro"),
#                 html.P(['Next year, DonorsChoose.org expects to receive close to 500,000 project proposals. As a result, there are three main problems they need to solve',html.Br(),'1. How to scale current manual processes and resources to screen 500,000 projects so that they can be posted as quickly and as efficiently as possible',
#                 html.Br(),'2. How to increase the consistency of project vetting across different volunteers to improve the experience for teachers', html.Br(),'3. How to focus volunteer time on the applications that need the most assistance'],className="Intro"),
                

#         ],fluid=True)
# #fig1 donut.....................................................................................................
# y_value_counts = df['project_is_approved'].value_counts()   
# data = [y_value_counts[1], y_value_counts[0]]
# labels=['Approved','Not Approved']
# colors=['#f89b9b','#f8bc98']
# layout =go.Layout()
# fig1_donut=go.Figure(data=[go.Pie(labels=labels,values=data,hole=.4,marker=dict(colors=colors))],layout=layout)
# #..............................................................................................................
# #fig2 us map..........................................................................................................
# temp = pd.DataFrame(df.groupby("school_state")["project_is_approved"].apply(np.mean)).reset_index()
# temp.columns = ['state_code', 'num_proposals']
# scl=[[0.0,'#A2E2F8'],[0.25,'#75BDE0'],[0.75,'#4A8DB7'],[1.0,'#3B7097']]
# data1 = [ dict(
#         type='choropleth',
#         colorscale = scl,
#         autocolorscale = False,
#         locations = temp['state_code'],
#         z = temp['num_proposals'].astype(float),
#         locationmode = 'USA-states',
#         text = temp['state_code'],
#         marker = dict(line = dict (color = 'rgb(255,255,255)',width = 2)),
#         colorbar = dict(title = "% of proposals")
#       ) ]

# layout1 = dict(
#         geo = dict(
#             scope='usa',
#             projection=dict( type='albers usa' ),
#             showlakes = True,
#             lakecolor = 'rgb(255, 255, 255)',
#         ),
#     )

# fig2_usmap = go.Figure(data=data1, layout=layout1)
# #..............................................................................................................
# #fig3 bar.....................................................................................................
# temp1 = pd.DataFrame(df.groupby("school_state")["project_is_approved"].agg(lambda x: x.eq(1).sum())).reset_index()
# temp1['total'] = pd.DataFrame(df.groupby('school_state')['project_is_approved'].agg(total='count')).reset_index()['total']
# temp1['Avg'] = pd.DataFrame(df.groupby('school_state')['project_is_approved'].agg(Avg='mean')).reset_index()['Avg']
# temp1.sort_values(by=['total'],inplace=True, ascending=False)
# trace1=go.Bar(x=temp1['school_state'],y=temp1['total'],name='Total',marker={'color':'#345DA7'})
# trace2=go.Bar(x=temp1['school_state'],y=temp1['project_is_approved'],name='Approved',marker={'color':'#4BB4DE'})
# layout=go.Layout(template='simple_white')
# data=[trace1,trace2]
# fig3_bar=go.Figure(data=data,layout=layout)
# #.................................................................................................................


# tab1_content = dbc.Container([
#     dbc.Row([
#         dbc.Col([
#             dbc.Card([
#                 dbc.CardHeader('Project status'),
#                 dbc.CardBody([dcc.Graph(figure=fig1_donut)]) 
#              ])
#          ],width={'size':4}),
#         dbc.Col([
#             dbc.Card([
#                 dbc.CardHeader('Project Proposals Percentage of Acceptance Rate by US States'),
#                 dbc.CardBody([dcc.Graph(figure=fig2_usmap)]) 
#              ])

#          ])

#      ]),
#      dbc.Row([
#          dbc.Col([
#              dbc.Card([
#                 dbc.CardHeader('State Total vs Approved'),
#                 dbc.CardBody([dcc.Graph(figure=fig3_bar)]) 
#              ])

#           ])
#       ],style={'margin-top':'1%'}),
#       dbc.Row([
#         dbc.Col([
#              dbc.Card([
#                 dbc.CardHeader('word cloud'),
#                 dbc.CardBody([html.Img(src='/assets/cld.png',width='100%',className='wordcloud')],className='img-fluid')
#                 ]) 
            

#           ],width=7), 
#         dbc.Col([
#              dbc.Card([
#                 dbc.CardHeader('Observations'),
#                 dbc.CardBody([html.P("modeBarButtonsToRemove (list; optional): Remove mode bar button by name. All modebar button names at https://github.com/plotly/plotly.js/blob/master/src/components/modebar/buttons.js Common names include: sendDataToCloud; (2D) zoom2d, pan2d, select2d, lasso2d, zoomIn2d, zoomOut2d, autoScale2d, resetScale2d; (Cartesian) hoverClosestCartesian, hoverCompareCartesian; (3D) zoom3d, pan3d, orbitRotation, tableRotation, handleDrag3d, resetCameraDefault3d, resetCameraLastSave3d, hoverClosest3d; (Geo) zoomInGeo, zoomOutGeo, resetGeo, hoverClosestGeo; hoverClosestGl2d, hoverClosestPie, toggleHover, resetViews.")],className='row3page5') 
#              ])

#           ],width=5) 

#       ],style={'margin-top':'1%'},className="cardfive_3rdrow")  

#  ],fluid=True,className="mt-3")
#  # tab 1...........................................................................................................

#  #tab 2............................................................................................................
# data=df
# y = data['project_is_approved'].values
# data.drop(['project_is_approved'], axis=1, inplace=True)
# X=data

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y)
# X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.33, stratify=y_train)

# features=[]

# vectorizer = CountVectorizer(min_df=10)
# vectorizer.fit(X_train['essay'].values)

# X_train_bow=vectorizer.transform(X_train['essay'].values)
# X_test_bow=vectorizer.transform(X_test['essay'].values)
# X_cv_bow=vectorizer.transform(X_cv['essay'].values)
# features.extend(vectorizer.get_feature_names())
# A=str(X_train_bow.shape)
# B=str(X_test_bow.shape)
# C=str(X_cv_bow.shape)

# vectorizer=TfidfVectorizer(min_df=10)
# vectorizer.fit(X_train['essay'].values)
# X_train_tfidf=vectorizer.transform(X_train['essay'].values)
# X_test_tfidf=vectorizer.transform(X_test['essay'].values)
# X_cv_tfidf=vectorizer.transform(X_cv['essay'].values)
# D=str(X_train_tfidf.shape)
# E=str(X_test_tfidf.shape)
# F=str(X_cv_tfidf.shape)

# from sklearn.preprocessing import Normalizer
# normalizer = Normalizer()
# normalizer.fit(X_train['price'].values.reshape(-1,1))
# X_train_price_norm = normalizer.transform(X_train['price'].values.reshape(-1,1))
# X_cv_price_norm = normalizer.transform(X_cv['price'].values.reshape(-1,1))
# X_test_price_norm = normalizer.transform(X_test['price'].values.reshape(-1,1))
# features.extend(X_train['price'])
# G=str(X_train_price_norm.shape)
# H=str(y_train.shape)
# I=str(X_cv_price_norm.shape)
# J=str(y_cv.shape)
# K=str(X_test_price_norm.shape)
# L=str(y_test.shape)

# from sklearn.preprocessing import Normalizer
# normalizer = Normalizer()
# normalizer.fit(X_train['teacher_number_of_previously_posted_projects'].values.reshape(-1,1))
# X_train_ppp_norm = normalizer.transform(X_train['teacher_number_of_previously_posted_projects'].values.reshape(-1,1))
# X_cv_ppp_norm = normalizer.transform(X_cv['teacher_number_of_previously_posted_projects'].values.reshape(-1,1))
# X_test_ppp_norm = normalizer.transform(X_test['teacher_number_of_previously_posted_projects'].values.reshape(-1,1))
# features.extend(X_train['teacher_number_of_previously_posted_projects'])
# M=str(X_train_ppp_norm.shape) 
# N=str(y_train.shape)
# O=str(X_cv_ppp_norm.shape)
# P=str(y_cv.shape)
# Q=str(X_test_ppp_norm.shape)
# R=str(y_test.shape)

# vectorizer = CountVectorizer()
# vectorizer.fit(X_train['teacher_prefix'].values) 
# X_train_teacher_ohe = vectorizer.transform(X_train['teacher_prefix'].values)
# X_cv_teacher_ohe = vectorizer.transform(X_cv['teacher_prefix'].values)
# X_test_teacher_ohe = vectorizer.transform(X_test['teacher_prefix'].values)
# features.extend(vectorizer.get_feature_names())

# vectorizer = CountVectorizer()
# vectorizer.fit(X_train['school_state'].values)
# X_train_state_ohe = vectorizer.transform(X_train['school_state'].values)
# X_cv_state_ohe = vectorizer.transform(X_cv['school_state'].values)
# X_test_state_ohe = vectorizer.transform(X_test['school_state'].values)

# features.extend(vectorizer.get_feature_names())

# vectorizer = CountVectorizer()
# vectorizer.fit(X_train['project_grade_category'].values)
# X_train_grade_ohe = vectorizer.transform(X_train['project_grade_category'].values)
# X_cv_grade_ohe = vectorizer.transform(X_cv['project_grade_category'].values)
# X_test_grade_ohe = vectorizer.transform(X_test['project_grade_category'].values)

# features.extend(vectorizer.get_feature_names())

# vectorizer = CountVectorizer()
# vectorizer.fit(X_train['clean_categories'].values) 
# X_train_cc_ohe = vectorizer.transform(X_train['clean_categories'].values)
# X_cv_cc_ohe = vectorizer.transform(X_cv['clean_categories'].values)
# X_test_cc_ohe = vectorizer.transform(X_test['clean_categories'].values)

# features.extend(vectorizer.get_feature_names())

# vectorizer = CountVectorizer()
# vectorizer.fit(X_train['clean_subcategories'].values) 
# X_train_csc_ohe = vectorizer.transform(X_train['clean_subcategories'].values)
# X_cv_csc_ohe = vectorizer.transform(X_cv['clean_subcategories'].values)
# X_test_csc_ohe = vectorizer.transform(X_test['clean_subcategories'].values)

# features.extend(vectorizer.get_feature_names())

# from scipy.sparse import hstack
# x_tr=hstack((X_train_bow,X_train_cc_ohe,X_train_csc_ohe,X_train_grade_ohe,X_train_ppp_norm,X_train_price_norm,
#              X_train_state_ohe,X_train_teacher_ohe)).tocsr()
# x_te=hstack((X_test_bow,X_test_cc_ohe,X_test_csc_ohe,X_test_grade_ohe,X_test_ppp_norm,X_test_price_norm,
#              X_test_state_ohe,X_test_teacher_ohe)).tocsr()
# xcv=hstack((X_cv_bow,X_cv_cc_ohe,X_cv_csc_ohe,X_cv_grade_ohe,X_cv_ppp_norm,X_cv_price_norm,
#            X_cv_state_ohe,X_cv_teacher_ohe)).tocsr()


# from sklearn.model_selection import GridSearchCV
# from sklearn.naive_bayes import MultinomialNB
# Mnb = MultinomialNB(class_prior=[0.5,0.5])
# parameters = {'alpha':[0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]}
# clf = GridSearchCV(Mnb, parameters, cv= 10, scoring='roc_auc',return_train_score=True)
# clf.fit(x_tr, y_train)
# train_auc= clf.cv_results_['mean_train_score']
# train_auc_std = clf.cv_results_['std_train_score']
# cv_auc = clf.cv_results_['mean_test_score'] 
# cv_auc_std = clf.cv_results_['std_test_score']
# trace0= go.Scatter(x=parameters['alpha'], y=train_auc,mode='lines')
# trace1= go.Scatter(x=parameters['alpha'], y=cv_auc,mode='lines')
# trace2= go.Scatter(x=parameters['alpha'], y=train_auc,mode='markers')
# trace3= go.Scatter(x=parameters['alpha'], y=cv_auc,mode='markers')
# data=[trace0,trace1,trace2,trace3]
# layout=go.Layout(title='ERROR PLOTS',xaxis=dict(title="Alpha:hyperparameter"),yaxis=dict(title='AUC'),template='simple_white')
# fig1=go.Figure(data=data,layout=layout)


# def pred_prob(clf, data): 
#     y_pred = []
#     y_pred = clf.predict_proba(data)[:,1]
#     return y_pred

# best_alpha1=clf.best_params_['alpha']


# from sklearn.metrics import roc_curve, auc
# nb_bow = MultinomialNB(alpha = best_alpha1,class_prior = [0.5,0.5])
# nb_bow.fit(x_tr, y_train)
# # roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
# # not the predicted outputy_train_pred = pred_prob(nb_bow,x_tr)
# y_test_pred = pred_prob(nb_bow,x_te)
# y_train_pred = pred_prob(nb_bow,x_tr)

# train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)
# test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred)
# trace4=go.Scatter(x=train_fpr,y=train_tpr,mode='lines')
# trace5=go.Scatter(x=test_fpr,y=test_tpr,mode='lines')
# data=[trace4,trace5]
# layout=go.Layout(title='AUC Curve',xaxis=dict(title='False Positive Rate(FPR)'),yaxis=dict(title='True Positive Rate(TPR)'),template='plotly_white')
# fig2=go.Figure(data=data,layout=layout)


# def find_best_threshold(threshold, fpr, tpr):
#     t = threshold[np.argmax(tpr*(1-fpr))]
#     # (tpr*(1-fpr)) will be maximum if your fpr is very low and tpr is very high
#     print("the maximum value of tpr*(1-fpr)", max(tpr*(1-fpr)), "for threshold", np.round(t,3))
#     return t

# def predict_with_best_t(proba, threshold):
#     predictions = []
#     for i in proba:
#         if i>=threshold:
#             predictions.append(1)
#         else:
#             predictions.append(0)
#     return predictions

# best_t = find_best_threshold(tr_thresholds, train_fpr, train_tpr)

# # print("Train data confusion matrix")
# # confusion_matrix_df_train = pd.DataFrame(confusion_matrix(y_train, predict_with_best_t(y_train_pred, best_t)), range(2),range(2))
# # sns.set(font_scale=1.4)#for label size
# # sns.heatmap(confusion_matrix_df_train, annot=True,annot_kws={"size": 16}, fmt='g')

# # print("Test data confusion matrix")
# # confusion_matrix_df_test = pd.DataFrame(confusion_matrix(y_test, predict_with_best_t(y_test_pred, best_t)), range(2),range(2))
# # sns.set(font_scale=1.4)#for label size
# # sns.heatmap(confusion_matrix_df_test, annot=True,annot_kws={"size": 16}, fmt='g')
# #design..................................................................................
# training=html.H6("Select Training Data Size Percentage:")

# radio_items=dcc.Dropdown(
#     options=[
#         {'label': '66%', 'value': 0.33},
#         {'label': '70%', 'value': 0.30},
#         {'label': '75%', 'value': 0.25},
#         {'label': '80%', 'value': 0.20}
#     ],
#     value=33
# ) 

# button=dbc.Button("SUBMIT", outline=True, color="primary", className="mr-1")
# card_countVEC=dbc.Card([
#     dbc.CardHeader("CountVectorizer"),
#     dbc.CardBody([
#         html.P(["Shape of Train matrix after one hot encodig :",A]),
#         html.P(["Shape of Test matrix after one hot encodig :",B]),
#         html.P(["Shape of Cross validation matrix after one hot encodig :",C])
#     ])

# ])

# card_tfidf=dbc.Card([
#     dbc.CardHeader("tfidfVectorizer"),
#     dbc.CardBody([
#         html.P(["Shape of Train matrix after one hot encodig :",D]),
#         html.P(["Shape of Test matrix after one hot encodig :",E]),
#         html.P(["Shape of Cross validation matrix after one hot encodig :",F])
#     ])

# ])

# price=dbc.Card([
#     dbc.CardHeader("price"),
#     dbc.CardBody([
#         html.P(["Shape of Train matrix after one hot encodig :",G,H]),
#         html.P(["Shape of Test matrix after one hot encodig :",I,J]),
#         html.P(["Shape of Cross validation matrix after one hot encodig :",K,L])
#     ])

# ])

# teacher_num=dbc.Card([
#     dbc.CardHeader("teacher_number_of_previously_posted_projects"),
#     dbc.CardBody([
#         html.P(["Shape of Train matrix after one hot encodig :",M,N]),
#         html.P(["Shape of Test matrix after one hot encodig :",O,P]),
#         html.P(["Shape of Cross validation matrix after one hot encodig :",Q,R])
#     ])

# ])

#  #tab2..........................................................................................................

# tab2_content = dbc.Card(
#     dbc.CardBody(
#         [
#             dbc.Row([
#                 dbc.Col([training]),
#                 dbc.Col([radio_items]),
#                 dbc.Col([button])
#             ]),
#             dbc.Row([
#                 dbc.Col([
#                     dbc.Card([
#                         dbc.CardHeader("Making Data Model Ready: encoding eassay"),
#                         dbc.CardBody([
#                             dbc.Row([
#                                 dbc.Col([card_countVEC],width=6),
#                                 dbc.Col([card_tfidf],width=6)
#                             ])
                            
#                         ]) 
#                     ])
#                 ])
#             ],style={'margin-top':'1%'}),
#             dbc.Row([
#                  dbc.Col([
#                     dbc.Card([
#                         dbc.CardHeader("Making Data Model Ready: encoding numerical, categorical features"),
#                         dbc.CardBody([
#                             dbc.Row([
#                                 dbc.Col([price]),
#                                 dbc.Col([teacher_num])
#                             ])
                            
#                         ]) 
#                     ])
#                 ])

#             ],style={'margin-top':'1%'}),
#             dbc.Row([
#                 dbc.Col([
#                     dcc.Graph(figure=fig1),
#                 ]),
#                 dbc.Col([
#                     dcc.Graph(figure=fig2)
#                 ])
#             ],style={'margin-top':'1%'})
#         ]   
#     ),
#     className="mt-3"
# )

# tabs = dbc.Tabs(
#     [
#         dbc.Tab(tab1_content, label="Data Analysis"),
#         dbc.Tab(tab2_content, label="Machine Learning"),
        
        
#     ]
# )




# output = html.Div(id='output',
#                   children=[],
#                   )

# def App_five():
#     layout = html.Div([
#         nav,
#         Intro,
#         tabs 
#     ])
#     return layout
