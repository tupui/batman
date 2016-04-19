import numpy as np
import pod


print "WTF"
class UQ:
    def __init__(self, settings):
        self.method = settings['method']
        self.points = settings['points']
    
    def sobol(self):
        print "Method: ", self.method
        print "Points: ", self.points
