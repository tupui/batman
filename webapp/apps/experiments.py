import os
import shutil
import sys
import json
import tempfile
import time
import batman.ui

from dash.dependencies import (Input, Output, State)
import dash_core_components as dcc
import dash_html_components as html

from app import app


class Semaphore:
    """Semaphore class.

    Using a tmp file, define a lock status for a running application.
    """

    def __init__(self):
        """Create tmp file to store a lock and set to unlock."""
        self._tmp = tempfile.TemporaryDirectory()
        workdir = self._tmp.name
        self.filename = os.path.join(workdir, 'semaphore.txt')
        with open(self.filename, 'w') as f:
            f.write('done')

    def lock(self):
        """Set a lock."""
        with open(self.filename, 'w') as f:
            f.write('working')

    def unlock(self):
        """Release the lock."""
        with open(self.filename, 'w') as f:
            f.write('done')

    def is_locked(self):
        """Verify lock status."""
        return open(self.filename, 'r').read() == 'working'


SEMAPHORE = Semaphore()

layout = html.Div([
    html.Button('Settings Incomplete',
                style={'width' : '100%', 'background-color': '#FF851B',
                       'border': 'none', 'border-radius': '0px',
                       'color': 'white', 'cursor': 'default'},
                id='settings_check'),

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
            children=html.Button(id='launch_button',
                                 className='button-primary'),
            id='confirm_launch',
            message='Danger danger! Are you sure you want to continue?',
        )
    ]),
    html.Div([
        html.Div(id='batman_running', style={'display': 'none'}),
        html.Div(children=False, id='batman_launch'),
        dcc.Interval(id='interval-component', interval=1500, disabled=False),  # in milliseconds
        html.Div(id='batman_log'),
    ], className='_dash-loading-callback_batman-running'),
])


@app.callback(Output('settings_check', 'style'),
              [Input('settings_status', 'children')],
              [State('settings_check', 'style')])
def settings_check(ready, style):
    if ready:
        style['display'] = 'none'
    return style


@app.callback(Output('launch_button', 'disabled'),
              [Input('settings_status', 'children'),
               Input('case_fname', 'value')])
def launch_button_state(settings_status, case_fname):
    """Disable launch button. If return True then button is disabled."""
    try:
        folder_exists = os.path.isdir(case_fname)
    except TypeError:
        folder_exists = False
    return (not settings_status) or (not folder_exists) or SEMAPHORE.is_locked()


@app.callback(Output('launch_button', 'style'),
              [Input('launch_button', 'disabled')])
def launch_button_style(button_disabled):
    """Button opacity if launch button not enable."""
    if button_disabled:
        return {'opacity': '0.3',  'cursor': 'not-allowed'}


@app.callback(Output('launch_button', 'children'),
              [Input('batman_running', 'children')])
def launch_button_text(running):
    """Button appearance."""
    out = 'Simulate Experiments'

    if running:
        out = [html.I(className='fa fa-circle-o-notch fa-spin'), ' ' + out]

    return out


@app.callback(Output('batman_launch', 'children'),
              [Input('confirm_launch', 'submit_n_clicks'),
               Input('cli_options', 'values'),
               Input('output_fname', 'value'),
               Input('case_fname', 'value')],
              [State('settings', 'children'),
               State('uuid', 'children')])
def simulate_experiments(submit_n_clicks, options, output_fname, case_fname, settings, uuid):
    """Execture BATMAN using defined settings and options."""
    if submit_n_clicks is not None:
        # prevent multiple batman launch
        if SEMAPHORE.is_locked():
            raise Exception('Resource is locked')
        else:
            SEMAPHORE.lock()

        try:
            shutil.rmtree('batman.log')
        except OSError:
            pass

        os.chdir(case_fname)
        output_fname = 'output' if output_fname is None else output_fname
        if 'force' in options:
            try:
                shutil.rmtree(output_fname)
            except OSError:
                pass

        settings_fname = 'settings-' + uuid + '.json'

        sys.argv = ['batman', settings_fname, '-o', output_fname]
        if 'no_surrogate' in options:
            sys.argv.append('-n')

        if 'uq' in options:
            sys.argv.append('-u')

        if 'quality' in options:
            sys.argv.append('-q')

        if 'restart' in options:
            sys.argv.append('-r')

        print(f'Launch options: {sys.argv}')

        with open(settings_fname, 'w') as fd:
            json.dump(json.loads(settings), fd)

        print(f'Settings feed to BATMAN:\n{settings}')

        # calling BATMAAAAN tu dudu du dudu
        i_time = time.time()
        batman.ui.main()

        SEMAPHORE.unlock()

        return f'BATMAN caught the Joker in {time.time() - i_time}s.'


@app.callback(Output('batman_running', 'children'),
              [Input('interval-component', 'n_intervals'),
               Input('confirm_launch', 'submit_n_clicks')])
def running_status(n , submit_n_clicks):
    """Running status."""
    try:
        running = SEMAPHORE.is_locked()
    except FileNotFoundError:
        running = False

    return True if (submit_n_clicks is not None) and running else False


# @app.callback(Output('interval-component', 'disabled'),
#               [Input('batman_running', 'children')])
# def interval_state(running):
#     """Start interval only if computation can start."""
#     return False if running else True


@app.callback(Output('batman_log', 'children'),
              [Input('batman_running', 'children')],
              [State('case_fname', 'value')])
def batman_log(running, case_fname):
    if running:
        return html.Div('BATMAN is busy chasing the Joker...',
                        style={'color': '#AAAAAA'})
    # try:
    #     return html.Iframe(src=os.path.join('batman.log'))
    # except TypeError:
    #     pass
