from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

from app import app
from apps import (settings, experiments)

app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})

STYLE_TABS = {'margin': '2% 3%'}

app.layout = html.Div([

    # header
    html.Div([
        html.Span('Uncertainty Quantification App using BATMAN API',
                  className='app-title', style={'font-size': '3.0rem'}),
        html.Div(
            html.Img(src='/assets/BatmanLogo.png', height='100%'),
            style={'float': 'right', 'height': '100%'})
    ], className='row header', style={'height': '50'}),

    # tabs
    html.Div([
        dcc.Tabs(
            id='tabs',
            style={'width': '100%',  #'height': '30',
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
    html.Div(settings.layout, id='settings_tab',
             className='row', style={'display': 'block'}),
    html.Div(experiments.layout, id='experiments_tab',
             className='row', style={'display': 'none'}),
    html.Div('Analysis', id='analysis_tab',
             className='row', style={'display': 'none'}),

])


@app.callback(Output('settings_tab', 'style'), [Input('tabs', 'value')])
def display_content_settings(tab):
    """Generate content for tabs."""
    STYLE_TABS['display'] = 'block' if tab == 'settings_tab' else 'none'
    return STYLE_TABS


@app.callback(Output('experiments_tab', 'style'), [Input('tabs', 'value')])
def display_content_experiments(tab):
    """Generate content for tabs."""
    STYLE_TABS['display'] = 'block' if tab == 'experiments_tab' else 'none'
    return STYLE_TABS


@app.callback(Output('analysis_tab', 'style'), [Input('tabs', 'value')])
def display_content_analysis(tab):
    """Generate content for tabs."""
    STYLE_TABS['display'] = 'block' if tab == 'analysis_tab' else 'none'
    return STYLE_TABS


if __name__ == '__main__':
    app.run_server(debug=True, threaded=False)
