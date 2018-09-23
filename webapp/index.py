import base64
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

import plotly.graph_objs as go
import plotly.tools as tls

from app import app
from apps import (settings, experiments)

app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})

app.layout = html.Div([

    # header
    html.Div([
        html.Span('Uncertainty Quantification App using BATMAN API',
                  className='app-title', style={'font-size': '3.0rem'}),
        html.Div(
            html.Img(src='/assets/BatmanLogo.png', height='100%'),
            style={'float': 'right', 'height': '100%'})
    ], className='row header', style={'height':'50'}),

    # tabs
    html.Div([
        dcc.Tabs(
            id='tabs',
            style={'width': '100%', #'height': '30',
                   'margin-top': '1em', 'margin-bottom': '1em'},
            children=[
                dcc.Tab(label='Settings', value='settings_tab'),
                dcc.Tab(label='Experiments', value='experiments_tab'),
                dcc.Tab(label='Analysis', value='analysis_tab'),
            ],
            value='settings_tab',
        )

    ], className='row tabs_div'),

    # tabs content
    html.Div(id='tab_content', className='row', style={'margin': '2% 3%'})

])


@app.callback(Output('tab_content', 'children'), [Input('tabs', 'value')])
def render_content(tab):
    """Generate content for tabs."""
    if tab == 'settings_tab':
        return settings.layout
    elif tab == 'experiments_tab':
        return experiments.layout
    else:
        return 'Analysis'


if __name__ == '__main__':
    app.run_server(debug=True, threaded=False)
