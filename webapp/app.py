import dash

app = dash.Dash(meta_tags=[
    {
        'name': 'description',
        'content': 'BATGIRL'
    }
])

app.title = 'Batgirl'

server = app.server
app.config.suppress_callback_exceptions = True
