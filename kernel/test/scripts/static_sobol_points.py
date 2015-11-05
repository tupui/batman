from static_sobol import *

import os
import sys

path = os.path.dirname(os.path.realpath(__file__))
paths = [os.path.join(path, '../output/static_sobol'),
         os.path.join(path, '../../src')]
for p in paths:
    if p not in sys.path:
        sys.path.insert(0, p)

from space import SpaceBase
s = SpaceBase()
s.read('output/static_sobol/pod/points.pickle')
space['provider'] = s
