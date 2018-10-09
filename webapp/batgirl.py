import uuid
from dash.dependencies import (Input, Output)
import dash_core_components as dcc
import dash_html_components as html

from app import app
from apps import (settings, experiments)

app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})

# # Loading screen CSS
# app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/brPBPO.css"})

# Loading button CSS
app.css.append_css({"external_url": "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"})

STYLE_TABS = {'margin': '2% 3%'}

app.layout = html.Div([

    # header
    html.Div([
        html.Div(html.Img(src='/assets/BatmanLogo.png', height='100%'),
                 className='title-logo'),
        html.Span('Batgirl', style={'color': 'var(--bat-yellow)'}),
        html.Span(' Uncertainty Quantification App using Batman API',
                  style={'color': 'white'}),
        html.Span(html.A(html.Button('View on GitLab',
                                     className='two columns title-button'),
                         href='https://gitlab.com/cerfacs/batman'))
    ], id='header', className="title"),

    # Unique identification of the session
    html.Div(id='uuid', style={'display': 'none'}),

    # tabs
    html.Div([
        dcc.Tabs(
            id='tabs',
            style={'width': '40%',
                   'margin-bottom': '1em',
                   'font-weight': 'bold',
                   'font-size': '1.5rem',
                   },
            colors={'primary': 'var(--bat-yellow)', 'background': '#F9FAFB'},
            children=[
                dcc.Tab(label='Settings', value='settings_tab'),
                dcc.Tab(label='Experiments', value='experiments_tab'),
                dcc.Tab(label='Analysis', value='analysis_tab'),
            ],
            value='settings_tab'
        )

    ], className='row'),

    # tabs content
    html.Div(settings.layout, id='settings_tab',
             className='row', style={'display': 'block'}),
    html.Div(experiments.layout, id='experiments_tab',
             className='row', style={'display': 'none'}),
    html.Div('Analysis', id='analysis_tab',
             className='row', style={'display': 'none'}),

    # html.Div(['© Copyright 2018, CERFACS - CECILL-B Licensed.'],
    #          style={'background-color':'#F012BE'})

])


@app.callback(Output('uuid', 'children'), [Input('header', 'children')])
def uuid_session(*args):
    """Create unique identifiant for each session."""
    return str(uuid.uuid4())


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
    app.run_server(debug=True, processes=True)
