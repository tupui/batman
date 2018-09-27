import os
import base64
import json
import tempfile
import copy

from dash.dependencies import (Input, Output)
import dash_core_components as dcc
import dash_html_components as html

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
            "method": "sobol",
            "distribution": [None]
        },
        "resampling": {
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
        "method": "kriging",
        "degree": 5,

    },
    "uq": {
        "sample": 1000,
        "test": "Ishigami",
        "pdf": ["Uniform(0, 1)"],
        "type": "aggregated",
        "method": "sobol"
    }
}


@app.callback(Output('settings_status', 'children'), [Input('settings', 'children')])
def validate_settings(settings):
    """Validate settings dict agains schema and update status."""
    _tmp = tempfile.TemporaryDirectory()
    workdir = _tmp.name
    settings = json.loads(settings)

    print(f'Settings being validated: {settings}')

    with open(os.path.join(workdir, 'settings.json'), 'w') as fd:
        json.dump(settings, fd)

    path = os.path.dirname(os.path.realpath(__file__))

    try:
        settings = import_config(os.path.join(workdir, 'settings.json'),
                                 os.path.join(path, 'schema.json'))
        status = True
    except SyntaxError:
        status = False

    return status


def settings_layout(contents):
    """Generate form from settings."""
    try:  # load from file
        content_type, content_string = contents.split(',')
        settings = base64.b64decode(content_string).decode('utf-8')
        settings = json.loads(settings)

        print(f'Settings loaded: {settings}')
    except AttributeError:  # load from dict
        settings = contents

    print(f'Settings used for form rendering:\n{settings}')

    kind = settings['space']['sampling']['method']
    ns = settings['space']['sampling']['init_size']
    dists = settings['space']['sampling']['distribution']
    corners = settings['space']['corners']
    plabels = settings['snapshot']['plabels']
    n_parameters = len(plabels)
    dim_max = settings['pod']['dim_max']
    pod_tolerance = settings['pod']['tolerance']
    pod_type = settings['pod']['type']

    layout = [
        html.H6('SPACE'),
        html.Span([
            html.Div([    
                html.Div([
                    html.Label('Number of parameters:'),
                    dcc.Slider(
                        id='n_parameters',
                        min=1, max=MAX_PARAMETERS, step=1, value=n_parameters,
                        marks={i: i for i in range(1, MAX_PARAMETERS + 1)}
                    )
                ], style={'width': '50%'}),
                html.Div(
                    style={'margin-bottom': '2em'},
                    children=[
                        # MAX_PARAMETERS Div are created but display set to none
                        html.Div(
                            id=f'parameter_{i}_container',
                            style={**PARAMETERS_STYLES},
                            children=[
                                f'Parameter {i}:',

                                html.Div([
                                    html.Div(
                                        dcc.Input(
                                            id=f'parameter_{i}_name',
                                            placeholder='name...',
                                            type='text', size=6,
                                            value=plabels[i - 1] if i - 1 < n_parameters else None
                                        ), className='two columns'),
                                    html.Div(
                                        dcc.Input(
                                            id=f'parameter_{i}_min',
                                            placeholder='min value...',
                                            type='text', size=8,
                                            value=corners[0][i - 1] if i - 1 < n_parameters else None
                                        ), className='two columns'),
                                    html.Div(
                                        dcc.Input(
                                            id=f'parameter_{i}_max',
                                            placeholder='max value...',
                                            type='text', size=8,
                                            value=corners[1][i - 1] if i - 1 < n_parameters else None
                                        ), className='two columns'),
                                    html.Div(
                                        dcc.Input(
                                            id=f'parameter_{i}_dist',
                                            placeholder="distribution...",
                                            type='text', size=20,
                                            value=dists[i - 1] if i - 1 < n_parameters else None
                                        ), className='three columns')
                                ]),
                            ], className='row') for i in range(1, MAX_PARAMETERS + 1)]),
                html.Div([
                    html.Div([
                        html.Label('Sampling method:'),
                        dcc.Dropdown(
                            id='parameter_method',
                            options=[
                                {'label': 'Sobol', 'value': 'sobol'},
                                {'label': 'Halton', 'value': 'halton'},
                                {'label': 'Latin Hypercube', 'value': 'lhs'},
                                {'label': 'Optimized LHS', 'value': 'olhs'},
                                {'label': 'Saltelli', 'value': 'saltelli'},
                            ],
                            value=kind),
                    ], className='six columns'),
                    html.Div([
                        html.Label('Sampling size:'),
                        dcc.Input(
                            id='parameter_ns',
                            placeholder='n samples...',
                            type='number',
                            value=ns)
                    ], className='four columns')
                ], className='row', id='ns_sampling_method'),
            ], className='six columns'),
            html.Div(id='visu_sample', className='six columns'),
        ], className='row'),

        html.Hr(),

        html.H6([dcc.Checklist(options=[{'value': 'enabled_pod'}],
                               values=['enabled_pod'], id='enabled_pod',
                               className='one columns'),
                 'POD'], className='row'),
        html.Div([
            html.Div([html.Label('Max modes'),
                      dcc.Input(
                          id='pod_dim_max', size=10,
                          placeholder='Max num modes...',
                          type='number', value=dim_max)], className='two columns'),
            html.Div([html.Label('Filtering tolerance'),
                      dcc.Slider(min=0, max=1, step=0.05, value=pod_tolerance,
                                 marks={i / 100: i for i in range(0, 100, 20)},
                                 id='pod_tolerance'),
                      ], className='four columns'),
            html.Div([html.Label('Update strategy'),
                      dcc.Dropdown(
                          options=[{'label': 'Static', 'value': 'static'},
                                   {'label': 'Dynamic', 'value': 'dynamic'}],
                          value=pod_type, id='pod_type')],
                     className='four columns'),
        ], className='row', id='pod'),

        html.Hr(),

        html.H6('SNAPSHOT'),
        html.Div([
            html.Div([html.Label('Number of workers'),
                      dcc.Input(
                          id='max_workers', size=10,
                          placeholder='Number of workers...',
                          type='number', value=5)], className='two columns'),
        ], className='row'),

        html.Hr(),

        html.H6([dcc.Checklist(options=[{'value': 'enabled_surrogate'}],
                               values=['enabled_surrogate'], id='enabled_surrogate',
                               className='one columns'),
                 'SURROGATE'], className='row'),
        html.Div(
            html.Div([html.Label('Method'),
                      dcc.Dropdown(
                              options=[{'label': 'Gaussian Process', 'value': 'kriging'},
                                       {'label': 'Polynomial Chaos', 'value': 'pc'}],
                              value='kriging', id='surrogate_method')
            ], className='three columns'), className='row'),
        html.Div(id='surrogate_args'),

        html.Hr(),

        html.H6('VISUALIZATION'),
        html.Div([
        ], className='row'),

        html.Hr(),

        html.H6('UQ'),
        html.Div([
        ], className='row'),
    ]

    return layout


app.callback(Output('settings_layout', 'children'), [Input('upload-settings', 'contents')])(settings_layout)

layout = html.Div([

    html.Div([
        html.H6('SETTINGS FILE'),

        dcc.Upload([
            'Drag and Drop or ',
            html.A('Select a File')],
            id='upload-settings', accept='application/json',
            style={'width': '30%', 'height': '60px', 'lineHeight': '60px',
                   'borderWidth': '1px', 'borderStyle': 'dashed',
                   'borderRadius': '5px', 'textAlign': 'center'},
            style_active={'borderColor': 'var(--bat-pink)'}
        ),

        html.Hr(),

        # Invisible Div storring settings dict
        html.Div(children=json.dumps(SETTINGS), id='settings', style={'display': 'none'}),
        html.Div(children=False, id='settings_status', style={'display': 'none'})
    ], className='row'),

    html.Div(settings_layout(SETTINGS), className='row', id='settings_layout')
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

space_values = [Input('parameter_method', 'value'),
                Input('parameter_ns', 'value'),
                Input('n_parameters', 'value')]
for i in range(1, MAX_PARAMETERS + 1):
    space_values.extend([Input(f'parameter_{i}_name', 'value'),
                         Input(f'parameter_{i}_min', 'value'),
                         Input(f'parameter_{i}_max', 'value'),
                         Input(f'parameter_{i}_dist', 'value')])

pod_values = [Input('pod_dim_max', 'value'), Input('pod_tolerance', 'value'),
              Input('pod_type', 'value')]
snapshot_values = [Input('max_workers', 'value')]
surrogate_values = [Input('surrogate_method', 'value')]
visualization_values = []
uq_values = []


@app.callback(Output('settings', 'children'),
              [*space_values, *pod_values, *snapshot_values,
               *surrogate_values, *visualization_values, *uq_values])
def update_settings(*settings_values):
    """Update settings from change in parameter values.

    Values are stored in hidden Div settings.
    """
    kind = settings_values[0]
    ns = settings_values[1]
    n_parameters = settings_values[2]

    plabels = settings_values[3::4][:n_parameters]  # 4 parameters per parameters

    pmins = settings_values[4::4][:n_parameters]

    pmins = [float(p) for p in pmins if p is not None]
    pmaxs = settings_values[5::4][:n_parameters]
    pmaxs = [float(p) for p in pmaxs if p is not None]

    dists_ = settings_values[6::4][:n_parameters]
    dists = []
    for dist, pmin, pmax in zip(dists_, pmins, pmaxs):
        dists.append('Uniform(' + str(pmin) + ',' + str(pmax) + ')'
                     if dist is None else dist)

    settings = copy.deepcopy(SETTINGS)
    settings['space'] = {'sampling': {'method': kind, 'init_size': ns,
                                      'distribution': dists},
                         'corners': [pmins, pmaxs]}
    settings['snapshot']['plabels'] = plabels

    idx = MAX_PARAMETERS * 4 + 3  # skip all space values

    settings['pod']['dim_max'] = settings_values[idx]
    settings['pod']['tolerance'] = settings_values[idx + 1]
    settings['pod']['type'] = settings_values[idx + 2]

    idx += 3  # skip all pod values

    settings['snapshot']['max_workers'] = settings_values[idx]

    idx += 1  # skip all snapshot values

    settings['surrogate']['method'] = settings_values[idx]

    return json.dumps(settings)


@app.callback(Output('visu_sample', 'children'), [Input('settings', 'children')])
def update_space_visu(settings):
    """Generate DoE visualization from settings."""
    settings = json.loads(settings)

    kind = settings['space']['sampling']['method']
    ns = settings['space']['sampling']['init_size']
    dists = settings['space']['sampling']['distribution']
    plabels = settings['snapshot']['plabels']
    corners = settings['space']['corners']

    print(f'Labels: {plabels} - Ns = {ns} | Method: {kind} - Corners: {corners}')

    try:
        space = Space(corners=corners, sample=ns, plabels=plabels)
        space.sampling(kind=kind, dists=dists)

        _tmp = tempfile.TemporaryDirectory()
        workdir = _tmp.name
        fig, _ = doe(space, plabels=plabels, fname=os.path.join(workdir, 'DOE.pdf'))
        fig = tls.mpl_to_plotly(fig, resize=True)
        # fig['layout']['width'] = 500
        # fig['layout']['height'] = 500
        output = [dcc.Graph(figure=fig, style={'width': '80%', 'margin-right': '5%'})]
    except:  # Catching explicitly the exceptions causes crash...
        output = [html.Img(src='/assets/loading-cylon.svg',
                           style={'width': '256', 'height': '42'}),
                  html.Div('Fill in space settings to display parameter space...',
                           style={'color': '#AAAAAA'})]

    return [html.H6('Parameter space visualization'), *output]


# Surrogate
@app.callback(Output('surrogate_args', 'children'),
              [Input('surrogate_method', 'value')])
def surrogate_args(surrogate_method):
    if surrogate_method == 'kriging':
        args = [html.Div([dcc.Checklist(
                               options=[{'label': 'Noise', 'value': 'noise'},
                                        {'label': 'Global optimizer', 'value': 'global_optimizer'}],
                               values=[''], id='noise_optim')
                         ], className='two columns'),
                html.Div([html.Div([html.Label('Kernel'),
                          dcc.Input(id='kernel',
                                    placeholder="kernel from OpenTURNS: 'Matern(length_scale=0.5, nu=0.5)'...",
                                    type='text')], className='five columns'),
                         ], className='four columns')]
    elif surrogate_method == 'pc':
        args = [
            html.Div([
                html.Div([html.Label('Strategy'),
                          dcc.RadioItems(id='strategy',
                             options=[{'label': 'Quadrature', 'value': 'Quad'},
                                      {'label': 'Least square', 'value': 'LS'},
                                      {'label': 'Sparse LS', 'value': 'SparseLS'}],
                             value='Quad')], className='four columns'),
                html.Div([html.Label('Degree'),
                          dcc.Input(id='degree',
                                    placeholder="Polynomial degree N=(p+1)^dim...",
                                    type='number')], className='two columns'),
            ], className='row'),
            html.Div([
                html.Label('Sparse parameters'),
                html.Div([
                    html.Div([html.Label('Maximum terms'),
                              dcc.Input(id='max_considered_terms',
                                        type='number')], className='two columns'),
                    html.Div([html.Label('Most significant'),
                              dcc.Input(id='most_significant',
                                        type='number')], className='two columns'),
                    html.Div([html.Label('Degree'),
                              dcc.Input(id='significance_factor',
                                        type='text')], className='two columns'),
                    html.Div([html.Label('Hyperbolic factor'),
                              dcc.Input(id='hyper_factor',
                                        type='text')], className='two columns')])
            ], className='row')
        ]

    return html.Div(args, className='row')
