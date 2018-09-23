import os
import base64
import json
import tempfile
import copy

import dash
from dash.dependencies import Input, State, Output
import dash_core_components as dcc
import dash_html_components as html

import plotly.graph_objs as go
import plotly.tools as tls
from batman.space import Space
from batman.visualization import doe
from batman.misc import import_config

from app import app

MAX_PARAMETERS = 10
PARAMETERS_STYLES = {
    'display': 'none',
    'margin-bottom': '1em',
    'margin-top': '1em',
}

SETTINGS = {
    "space": {
        "corners": [
            [None],
            [None]
        ],
        "sampling": {
            "init_size": 10,
            "method": "sobol"
        },
        "resampling":{
            "delta_space": 0.08,
            "resamp_size": 0,
            "method": "sigma",
            "hybrid": [["sigma", 4], ["loo_sobol", 2]],
            "q2_criteria": 0.9
        }
    },
    "pod": {
        "dim_max": 100,
        "tolerance": 0.99,
        "type": "static"
    },
    "snapshot": {
        "max_workers": 10,
        "plabels": [None],
        "flabels": ["F"],
        "provider": {
            "type": "function",
            "module": "function",
            "function": "f"
        },
        "io": {
            "space_fname": "sample-space.json",
            "space_format": "json",
            "data_fname": "sample-data.npz",
            "data_format": "npz"
        }
    },
    "surrogate": {
        "method": "kriging"
    },
    "uq": {
        "sample": 1000,
        "test": "Ishigami",
        "pdf": ["Uniform(0, 1)"],
        "type": "aggregated",
        "method": "sobol"
    }
}


def space_layout(contents):
    """Generate form from settings."""
    try:
        content_type, content_string = contents.split(',')
        settings = base64.b64decode(content_string).decode('utf-8')
        settings = json.loads(settings)

        _tmp = tempfile.TemporaryDirectory()
        workdir = _tmp.name
        with open(os.path.join(workdir, 'settings.json'), 'w') as fd:
            json.dump(settings, fd)

        path = os.path.dirname(os.path.realpath(__file__))
        settings = import_config(os.path.join(workdir, 'settings.json'),
                                 os.path.join(path, 'schema.json'))

        print(f'Settings loaded: {settings}')
    except AttributeError:
        settings = contents

    print(f'Settings used for form rendering:\n{settings}')

    kind = settings['space']['sampling']['method']
    ns = settings['space']['sampling']['init_size']
    corners = settings['space']['corners']
    plabels = settings['snapshot']['plabels']
    n_parameters = len(plabels)

    layout = [
        html.Div([
            html.Div([
                html.Label('Number of parameters:'),
                dcc.Slider(
                    id='n_parameters',
                    min=1, max=MAX_PARAMETERS, step=1, value=n_parameters,
                    marks = {i: i for i in range(1, MAX_PARAMETERS + 1)}
                )
            ], style={'width': '50%'}),
            html.Div(
                style={'margin-bottom':'2em'},
                children=[
                    # MAX_PARAMETERS Div are created but display set to none
                    html.Div(
                        id=f'parameter_{i}_container',
                        style={**PARAMETERS_STYLES},
                        children=[
                            f'Parameter {i}:',

                            html.Div([
                                dcc.Input(
                                    id=f'parameter_{i}_name',
                                    placeholder='name...',
                                    type='text',
                                    value=plabels[i - 1] if i - 1 < n_parameters else None
                                ),
                                dcc.Input(
                                    id=f'parameter_{i}_min',
                                    placeholder='min value...',
                                    type='text',
                                    value=corners[0][i - 1] if i - 1 < n_parameters else None
                                ),
                                dcc.Input(
                                    id=f'parameter_{i}_max',
                                    placeholder='max value...',
                                    type='text',
                                    value=corners[1][i - 1] if i - 1 < n_parameters else None
                                )]
                            ),
                        ]) for i in range(1, MAX_PARAMETERS + 1)]),
            html.Div([
                html.Label('Sampling method:'),
                dcc.Dropdown(
                    id='parameter_method',
                    style={'width': '50%'},
                    options=[
                        {'label': 'Sobol', 'value': 'sobol'},
                        {'label': 'Halton', 'value': 'halton'},
                        {'label': 'Latin Hypercube', 'value': 'lhs'}
                    ],
                    value=kind
                ),
                html.Label('Sampling size:'),
                dcc.Input(
                    id='parameter_ns',
                    placeholder='n samples...',
                    type='number',
                    value=ns
                )

            ]),
        ], className='seven columns'), 

        html.Div(id='visu_sample', className='five columns',
                 style={'display': 'block'})
    ]

    return layout


app.callback(Output('space', 'children'), [Input('upload-settings', 'contents')])(space_layout)

layout = html.Div([

    html.Div([
        html.H3('Load Settings File'),

        dcc.Upload([
            'Drag and Drop or ',
            html.A('Select a File')],
            id='upload-settings', accept='application/json',
            style={
            'width': '30%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center'
        }),

        # Invisible Div storring settings dict
        html.Div(children=json.dumps(SETTINGS), id='settings', style={'display': 'none'})
    ], className='row'),

    html.H3('Space'),
    html.Div(space_layout(SETTINGS), className='row', id='space')
])


# Space
for n in range(1, MAX_PARAMETERS + 1):
    @app.callback(Output(f'parameter_{n}_container', 'style'),
                 [Input('n_parameters', 'value')])
    def callback(n_parameters, n=n):
        new_styles = {**PARAMETERS_STYLES}
        if n <= n_parameters:
            new_styles['display'] = 'block'
        else:
            new_styles['display'] = 'none'
        return new_styles

parameter_values = [Input('parameter_method', 'value'),
                    Input('parameter_ns', 'value'),
                    Input('n_parameters', 'value')]
for i in range(1, MAX_PARAMETERS + 1):
    parameter_values.extend([Input(f'parameter_{i}_name', 'value'),
                             Input(f'parameter_{i}_min', 'value'),
                             Input(f'parameter_{i}_max', 'value')])


@app.callback(Output('settings', 'children'), parameter_values)
def update_settings(*parameter_values):
    """Update settings from change in parameter values.

    Values are stored in hidden Div settings.
    """
    kind = parameter_values[0]
    ns = parameter_values[1]
    n_parameters = parameter_values[2]

    plabels = parameter_values[3::3][:n_parameters]
    pmins = parameter_values[4::3][:n_parameters]
    pmaxs = parameter_values[5::3][:n_parameters]

    settings = copy.deepcopy(SETTINGS)
    settings['space'] = {'samping': {'method': kind, 'init_size': ns},
                         'corners': [pmins, pmaxs]}
    settings['snapshot']['plabels'] = plabels

    return json.dumps(settings)


@app.callback(Output('visu_sample', 'children'), parameter_values)
def update_space_visu(*parameter_values):
    """Generate DoE visualization from settings."""
    kind = parameter_values[0]
    ns = parameter_values[1]
    n_parameters = parameter_values[2]

    plabels = parameter_values[3::3][:n_parameters]
    pmins = parameter_values[4::3][:n_parameters]
    pmaxs = parameter_values[5::3][:n_parameters]

    print(f'Labels: {plabels} - Ns = {ns} | Method: {kind} - Corners: {[pmins, pmaxs]}')

    try:
        pmins = [float(p) for p in pmins]
        pmaxs = [float(p) for p in pmaxs]
        space_settings = {'corners': [pmins, pmaxs], 'sample': ns,
                          'plabels': plabels}

        space = Space(corners=[pmins, pmaxs], sample=ns, plabels=plabels)
        space.sampling(kind=kind)

        fig, _ = doe(space, plabels=plabels, fname=os.path.join('.', 'DOE.pdf'))
        fig = tls.mpl_to_plotly(fig, resize=True)
        fig['layout']['width'] = 400
        fig['layout']['height'] = 400
        output = [dcc.Graph(figure=fig)]
    except:  # Catching explicitly the exceptions causes crash...
        output = [html.Img(src='/assets/loading.gif'),
                  html.Div('Fill in space settings to display parameter space...')]

    return [html.Label('Parameter space:'), *output]
