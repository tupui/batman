# encoding: utf-8
import os
from batman.input_output import formater
from tests.plugins import f_snapshot

coupling = 'coupling-dir'

input_formater = formater('json')
output_formater = formater('csv')

point = input_formater.read(os.path.join(coupling, 'sample-space.json'), ['X1', 'X2'])
result = f_snapshot(point)
output_formater.write(os.path.join(coupling, 'sample-data.csv'), result,
                      ['F1', 'F2', 'F3'], [1, 1, 2])
