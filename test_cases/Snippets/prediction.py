#!/usr/bin/env python
# coding:utf-8
"""Predictions.

Add prediction points within settings.json.
Addapt this script to your case.

"""
import numpy as np
import itertools
import json

num = 25
settings_path = './'

with open(settings_path + 'settings.json', 'r') as f:
    settings = json.load(f)

x = np.linspace(settings['space']['corners'][0][0],
                settings['space']['corners'][1][0], num=num)
y = np.linspace(settings['space']['corners'][0][1],
                settings['space']['corners'][1][1], num=num)

points = []
for i, j in itertools.product(x, y):
    points += [(float(i), float(j))]

settings['surrogate']['predictions'] = points

with open(settings_path + 'settings-prediction.json', 'w') as f:
    json.dump(settings, f, indent=4)
