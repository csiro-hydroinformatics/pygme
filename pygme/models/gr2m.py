
import math
import numpy as np
import pandas as pd

from hydrodiy.data.containers import Vector
from pygme.model import Model, ParamsVector
#from pygme.calibration import Calibration

import c_pygme_models_hydromodels
import c_pygme_models_utils



class GR2M(Model):

    def __init__(self):

        # Config vector
        config = Vector(['catcharea'], [0], [0])

        # params vector
        vect = Vector(['X1', 'X2'], \
                    [400, 0.8], [10., 0.1], [1e4, 3])
        params = ParamsVector(vect)

        # State vector
        states = Vector(['S', 'R'])

        # Model
        Model.__init__(self, 'GR2M',
            config, params, states, \
            ninputs=2, \
            noutputsmax=9)


    def initialise(self, states=None, uhs=None):
        ''' Initialise state vector and potentially all UH states vectors '''
        if states is None:
            X1, _ = self.params.values
            self.states.values = [X1/2, 30]
        else:
            self.states.values = states


    def run(self):
        ierr = c_pygme_models_hydromodels.gr2m_run(self.istart, self.iend,
            self.params.values, \
            self.inputs, \
            self.states.values, \
            self.outputs)

        if ierr > 0:
            raise ValueError(('Model gr2m, c_pygme_models_hydromodels.gr2m_run' + \
                    'returns {0}').format(ierr))



#class CalibrationGR2M(Calibration):
#
#    def __init__(self, timeit=False):
#
#        gr = GR2M()
#
#        Calibration.__init__(self,
#            model = gr,
#            warmup = 12,
#            timeit = timeit)
#
#        # Calibration on sse square root with bias constraint
#        self._errfun.constants = [0.5, 2., 1.]
#
#        self._calparams.means =  [5.9, -0.28]
#        self._calparams.covar = [[0.52, 0.015], [0.015, 0.067]]
#
#    def cal2true(self, calparams):
#        return np.exp(calparams)


