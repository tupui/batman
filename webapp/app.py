import os
import dash
from dash.dependencies import Input, State, Output
import dash_core_components as dcc
import dash_html_components as html

import plotly.graph_objs as go
# from plotly.offline import plot_mpl
import plotly.tools as tls
from batman.space import Space
from batman.visualization import doe


MAX_PARAMETERS = 10
PARAMETERS_STYLES = {
    'display': 'none',
    'margin-bottom': '1em',
    'margin-top': '1em',
}

app = dash.Dash()
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

app.layout = html.Div([
    html.H1('Space'),
    html.Label('Number of parameters:'),
    dcc.Slider(
        id='n_parameters',
        min=1, max=10, step=1, value=1,
        marks = {i: i for i in range(1, MAX_PARAMETERS + 1)}
    ),
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
                            type='text'
                        ),
                        dcc.Input(
                            id=f'parameter_{i}_min',
                            placeholder='min value...',
                            type='text'
                        ),
                        dcc.Input(
                            id=f'parameter_{i}_max',
                            placeholder='max value...',
                            type='text'
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
                {'label': 'Optimized Latin Hypercube', 'value': 'olhs'}
            ],
            value='sobol'
        ),
        html.Label('Sampling size:'),
        dcc.Input(
            id='parameter_ns',
            placeholder='n samples...',
            type='number'
        )

    ]),
    html.Label('Parameter space:'),
    # html.Div(id='visu_sample'),
    dcc.Graph(id='visu_sample', figure={}),
])

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

@app.callback(Output('visu_sample', 'figure'), parameter_values)
def update_sampling(*parameter_values):

    print(parameter_values)

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
        fig = tls.mpl_to_plotly(fig)
    except:  # Catching explicitly the exceptions causes crash...
        fig = {}

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
