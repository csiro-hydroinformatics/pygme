
import numpy as np
import pandas as pd

from hydrodiy.data.containers import Vector
from pygme.model import Model, ParamsVector, UH
#from pygme.calibration import Calibration

import c_pygme_models_hydromodels


class GR4J(Model):

    def __init__(self):

        # Config vector
        config = Vector(['continuous'],\
                    [0], [0], [1])

        # params vector
        vect = Vector(['X1', 'X2', 'X3', 'X4'], \
                    [400, -1, 50, 0.5], \
                    [10, -50, 1, 0.5], \
                    [2e4, 50, 5e3, 1e2])
        uhs = [UH('gr4j_ss1_daily', 3), UH('gr4j_ss2_daily', 3)]
        params = ParamsVector(vect, uhs)

        # State vector
        states = Vector(['S', 'R'])

        # Model
        Model.__init__(self, 'GR4J',
            config, params, states, \
            ninputs=2, \
            noutputsmax=9)

    def run(self):
        uh1 = self.params.uhs[0]
        uh2 = self.params.uhs[1]

        ierr = c_pygme_models_hydromodels.gr4j_run(uh1.nuh, \
            uh2.nuh, self.istart, self.iend, \
            self.params.values, \
            uh1.ord, \
            uh2.ord, \
            self.inputs, \
            uh1.states, \
            uh2.states, \
            self.states.values, \
            self.outputs)

        if ierr > 0:
            raise ValueError(('c_pygme_models_hydromodels.gr4j_run' +
                ' returns {0}').format(ierr))


#class CalibrationGR4J(Calibration):
#
#    def __init__(self, timeit=False):
#
#        gr = GR4J()
#
#        Calibration.__init__(self,
#            model = gr,
#            warmup = 365,
#            timeit = timeit)
#
#        # Calibration on sse square root with bias constraint
#        self._errfun.constants = [0.5, 2., 1.]
#
#        self._calparams.means =  [5.8, -0.78, 3.39, 0.86]
#
#        covar = [[1.16, 0.2, -0.15, -0.07],
#                [0.2, 1.79, -0.24, -0.149],
#                [-0.15, -0.24, 1.68, -0.16],
#                [-0.07, -0.149, -0.16, 0.167]]
#        self._calparams.covar = covar
#
#
#    def cal2true(self, calparams):
#        params = np.array([np.exp(calparams[0]),
#                np.sinh(calparams[1]),
#                np.exp(calparams[2]),
#                0.49+np.exp(calparams[3])])
#
#        return params
#

