import os
import itertools
from pyramid.view import view_config, view_defaults
import colander
import deform.widget
from plotly.offline import plot_mpl
from batman.space import Space
from batman.visualization import doe


class SchemaSampling(colander.Schema):
    init_size = colander.SchemaNode(
        colander.Int(),
        default=10,
        description='Number of simulation to perform')
    choices = (
            ('uniform', 'Uniform'),
            ('halton', 'Halton'),
            ('sobol', "Sobol'"),
            ('sobolscramble', "Sobol' Scrambled"),
            ('lhsc', 'LHS centered'),
            ('lhsr', 'LHS randomized'),
            ('faure', 'Faure')
            )
    method = colander.SchemaNode(
        colander.String(),
        widget=deform.widget.SelectWidget(values=choices),
        description='Method used to generate sample')


class SchemaHybrid(colander.TupleSchema):
    method = colander.SchemaNode(colander.String(),
                                 validator=colander.OneOf(
                                     ['sigma', 'loo_sigma', 'loo_sobol',
                                      'extrema', 'hybrid', 'discrepancy',
                                      'optimization']))
    number = colander.SchemaNode(colander.Int(),
                                 validator=colander.Range(min=0))


class SchemaHybridSeq(colander.SequenceSchema):
    hybrid = SchemaHybrid()


class SchemaResamplingMethod(colander.MappingSchema):
    choices = (
            ('sigma', 'Sigma (variance of the model)'),
            ('loo_sigma', 'Leave-One-Out Sigma'),
            ('loo_sobol', "Leave-One-Out Sobol'"),
            ('discrepancy', 'Discrepancy'),
            ('optimization', 'Optimization (expected improvement)'),
            ('extrema', 'Extrema'),
            ('hybrid', 'Hybrid'),
            )
    method = colander.SchemaNode(
        colander.String(),
        # default='sigma',
        missing=colander.drop,
        widget=deform.widget.SelectWidget(values=choices),
        description='Method used to generate sample')

    hybrid = SchemaHybridSeq(description='Only for hybrid method: [method, number]',
                             missing=colander.drop)


class SchemaResampling(colander.MappingSchema):
    delta_space = colander.SchemaNode(
        colander.Float(),
        # default=0.08,
        validator=colander.Range(0, 1),
        missing=colander.drop,
        description='Innerspace defined by corners for resampling')
    resamp_size = colander.SchemaNode(
        colander.Int(),
        validator=colander.Range(min=0),
        # default=0,
        missing=colander.drop,
        description='Number of simulation to perform for resampling')

    def hybrid_check(form, value):
        if (value['method'] == 'hybrid') and (not value['hybrid']):
            exc = colander.Invalid(
                    form, 'Hybrid strategy is required if resampling is hybrid')
            exc['hybrid'] = 'Required if method is hybrid'
            raise exc

    method = SchemaResamplingMethod(validator=hybrid_check)
    
    q2_criteria = colander.SchemaNode(
        colander.Float(),
        validator=colander.Range(0, 1),
        missing=colander.drop,
        # default=0.8,
        description='Quality stopping criteria')


class SchemaParameter(colander.Schema):
    name = colander.SchemaNode(colander.String())
    min_value = colander.SchemaNode(colander.Float())
    max_value = colander.SchemaNode(colander.Float())


class SchemaParameterSeq(colander.SequenceSchema):
    def min_max_validator(form, value):
        if value['max_value'] < value['min_value']:
            exc = colander.Invalid(form,
                                   'Max value must be greater than Min value')
            exc['max_value'] = 'Must be greater than Min value'
            raise exc
    parameters = SchemaParameter(validator=min_max_validator)


class SchemaSpace(colander.MappingSchema):
    parameters = SchemaParameterSeq()

    sampling = SchemaSampling(
        title="Sampling",
        widget=deform.widget.MappingWidget(
            template="mapping_accordion",
            open=True))

    resampling = SchemaResampling(
        title="Resampling",
        missing=colander.drop,
        widget=deform.widget.MappingWidget(
            template="mapping_accordion",
            open=False))


class SchemaSurrogate(colander.Schema):
    choices = (
            ('pc', 'Polynomial Chaos'),
            ('kriging', 'Kriging (Gaussian Process)'),
            ('rbf', "Radial Basis Functions"))
    method = colander.SchemaNode(
        colander.String(),
        missing=colander.drop,
        widget=deform.widget.SelectWidget(values=choices),
        description='Surrogate Model method')


counter = itertools.count()
forms = {'form1': {'form': None, 'captured': None},
         'form2': {'form': None, 'captured': None}}

schema1 = SchemaSpace(title='Space')
forms['form1']['form'] = deform.Form(schema1, buttons=('submit',),
                                     formid='form1',
                                     counter=counter)

schema2 = SchemaSurrogate(title='Surrogate')
forms['form2']['form'] = deform.Form(schema2, buttons=('submit',),
                                     formid='form2',
                                     counter=counter)

form_completed = {'form1': False, 'form2': False}

@view_defaults(route_name='settings')
class BatGirlViews(object):
    def __init__(self, request):
        self.request = request
        self.view_name = 'Settings Views'

    @view_config(route_name='home', renderer='templates/home.jinja2')
    def home(self):
        return {'page_title': 'Home View'}

    @view_config(renderer='templates/settings.jinja2')
    def settings(self):
        html = []
        output = {}

        if 'run_batman' in self.request.POST:
            print('Running batman')

        if 'submit' in self.request.POST:
            posted_formid = self.request.POST['__formid__']
            for (formid, form) in forms.items():
                if formid == posted_formid:
                    try:
                        controls = self.request.POST.items()
                        form['captured'] = form['form'].validate(controls)
                        html.append(form['form'].render(form['captured']))
                        form_completed[formid] = True
                    except deform.ValidationFailure as e:
                        # the submitted values could not be validated
                        html.append(e.render())
                else:
                    if form['captured'] is not None:
                        html.append(form['form'].render(form['captured']))
                    else:
                        html.append(form['form'].render())

            # Render graphs
            try:
                output.update(self.generate_space(forms['form1']['captured']))
            except TypeError:
                pass
        else:
            for _, form in forms.items(): 
                html.append(form['form'].render())

        reqts = forms['form1']['form'].get_widget_resources()

        html = ''.join(html)

        output.update({'form': html, 'reqts': reqts,
                       'form_completed': all(form_completed.values())})

        # values passed to template for rendering
        return output

    def generate_space(self, data):
        """Interpret settings to render space with plotly."""
        corners = [[], []]
        plabels = []
        for parameter in data['parameters']:
            corners[0].append(parameter['min_value'])
            corners[1].append(parameter['max_value'])
            plabels.append(parameter['name'])

        space_settings = {'corners': corners, 
                          'sample': data['sampling']['init_size'],
                          'plabels': plabels}

        space = Space(**space_settings)
        space.sampling(kind=data['sampling']['method'])

        disc = space.discrepancy()

        fig, _ = doe(space, fname=os.path.join('.', 'DOE.pdf'))

        script = plot_mpl(fig, auto_open=False, output_type='div',
                          resize=True, strip_style=True)

        return {'script_plotly': script, 'discrepancy': disc}
