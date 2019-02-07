
from dash.dependencies import (Input, Output)
import dash_core_components as dcc
import dash_html_components as html

import plotly.tools as tls
from batman.space import Space
# from batman.visualization import pairplot

from app import app


layout = html.Div([

    html.H3('Data Files'),
    # html.P('Enter absolute paths here.'),
    html.Div([
        html.Div([
            html.Label('Sample File'),
            dcc.Upload([
                'Drag and Drop or ',
                html.A('Select a File')],
                id='upload-sample', accept='application/json',
                style={'width': '40%', 'height': '60px', 'lineHeight': '60px',
                       'borderWidth': '1px', 'borderStyle': 'dashed',
                       'borderRadius': '5px', 'textAlign': 'center'},
                style_active={'borderColor': 'var(--bat-pink)'}
            ),
        ], className='six columns'),
        html.Div([
            html.Label('Data File'),
            dcc.Upload([
                'Drag and Drop or ',
                html.A('Select a File')],
                id='upload-data', accept='application/json',
                style={'width': '40%', 'height': '60px', 'lineHeight': '60px',
                       'borderWidth': '1px', 'borderStyle': 'dashed',
                       'borderRadius': '5px', 'textAlign': 'center'},
                style_active={'borderColor': 'var(--bat-pink)'}
            ),
        ], className='six columns'),
    ], className='row'),

    html.Hr(),

    html.H6('Pairplot'),



    ])


@app.callback(Output('sample', 'children'), [
             Input('upload-sample', 'contents')])
def sample():
    pass


@app.callback(Output('data', 'children'), [
             Input('upload-data', 'contents')])
def data():
    pass


# fig, _ = pairplot(space, data, plabels=plabels)
# fig = tls.mpl_to_plotly(fig, resize=True)
# output = [dcc.Graph(figure=fig, style={'width': '80%', 'margin-right': '5%'})]

