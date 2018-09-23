import os
import shutil
import sys
import json
import re
import copy
import batman.ui
import batman.misc

import dash
from dash.dependencies import Input, State, Output
import dash_core_components as dcc
import dash_html_components as html

from app import app

# from .settings import layout as layout_settings

# Check settings and paths before activating button
# settings = json.loads(settings)

# _tmp = tempfile.TemporaryDirectory()
# workdir = _tmp.name
# with open(os.path.join(workdir, 'settings.json'), 'w') as fd:
#     json.dump(settings, fd)

# path = os.path.dirname(os.path.realpath(__file__))
# settings = import_config(os.path.join(workdir, 'settings.json'),
#                          os.path.join(path, 'schema.json'))

layout = html.Div([
    html.H3('Settings Sanity Check'),
    # html.Div([]),
    html.H3('Some Paths'),
    html.P('Enter absolute paths here.'),
    html.Div([
        html.Div([
            html.Label('Output Folder'),
            dcc.Input(
                id='output_fname',
                placeholder='output folder path...',
                type='text',
                size=50
            ),
        ], className='six columns'),
        html.Div([
            html.Label('Case Folder'),
            dcc.Input(
                id='case_fname',
                placeholder='case folder path...',
                type='text',
                size=50
            )
        ], className='six columns'),
    ], className='row'),

    html.Div([
        html.H3('Some Options'),
        dcc.Checklist(
            options=[
                {'label': 'Clear all', 'value': 'force'},
                {'label': 'Restart', 'value': 'restart'},
                {'label': 'Quality', 'value': 'quality'},
                {'label': 'No Surrogate', 'value': 'no_surrogate'},
                {'label': 'UQ', 'value': 'uq'}
            ],
            id='cli_options',
            values=[''],
            labelStyle={'display': 'inline-block', 'margin-right': '1em'}
        )
    ]),

    html.Div([
        html.H3('Launch BATMAN'),
        dcc.ConfirmDialogProvider(
            children=html.Button(
                'Simulate Experiments',
            ),
            id='confirm_launch',
            message='Danger danger! Are you sure you want to continue?'
        ),

        html.Div(id='launch_batman')
    ]),
])


@app.callback(Output('launch_batman', 'children'),
              [Input('confirm_launch', 'submit_n_clicks'),
               Input('cli_options', 'values'),
               Input('output_fname', 'value'),
               Input('case_fname', 'value'),
               # Input('settings', 'children'),
               ])
def simulate_experiments(submit_n_clicks, options, output_fname, case_fname,
                         *settings):
    """Execture BATMAN using defined settings and options."""
    if submit_n_clicks is not None:
        if 'force' in options:
            try:
                shutil.rmtree(output_fname)
            except OSError:
                pass

        sys.argv = ['batman', 'settings.json', '-o', output_fname]
        if 'no_surrogate' in options:
            sys.argv.append('-n')

        if 'uq' in options:
            sys.argv.append('-u')

        if 'quality' in options:
            sys.argv.append('-q')

        if 'restart' in options:
            sys.argv.append('-r')

        print(sys.argv)

        # os.chdir(case_fname)
        # with open('settings.json', 'w') as fd:
        #     json.dump(settings, fd)

        print(f'Settings feed to BATMAN:\n{settings}')

        # calling BATMAN tu dudu du dudu
        # batman.ui.main()

        # bat_log = dcc.Textarea(
        #     value='This is where the log goes...',
        #     style={'width': '100%'}
        # )   

        return 'bat_log'
