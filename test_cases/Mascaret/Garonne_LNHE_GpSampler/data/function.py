#!/usr/bin/env python
# coding:utf-8

import numpy as np
from TelApy.tools.studyMASC_UQ import MascaretStudy
from batman.input_output import formater
import logging
logging.basicConfig(level=logging.INFO)

# Create an instance of MascaretStudy from a JSON settings file.
study = MascaretStudy('config_garonne_lnhe_gpsampler.json')

# Read the input data from an input file
io = formater('npy')
params = io.read('./batman-coupling/sample-space.npy', ['Ks', 'Q'])
Ks_param = params[0, 0]
Q_param = params[0, 1:]
Ks_value = float(Ks_param)
Q_value = [float(x) for x in Q_param]

# Convert the input data to the input format of MascaretStudy
Ks = [{'type': 'zone', 'index': 0, 'value': Ks_value}]
Q = [{'type': 'discharge', 'index': 0, 'value': Q_value}]

# Run the instance of Mascaret for these input data
hydraulic_state = study(x={'friction_coefficients': Ks,
                           'boundary_conditions': Q})

# Extract the curvilinear abscissa and the water level
curv_abs = hydraulic_state['s']
water_level = hydraulic_state['z']

# Write the output data into an output file
names = ['curvilinear_abscissa', 'water_level']
sizes = [np.size(curv_abs), np.size(water_level)]
data = np.append(curv_abs, water_level)
io.write('./batman-coupling/sample-data.npy', data, names, sizes)
