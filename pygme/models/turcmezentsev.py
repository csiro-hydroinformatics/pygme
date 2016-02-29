
import math
import numpy as np
import pandas as pd

from hystat import sutils

from pygme.model import Model
from pygme.calibration import Calibration


class TurcMezentsev(Model):

    def __init__(self,
            nens_params=1,
            nens_states=1,
            nens_outputs=1):


        Model.__init__(self, 'turcmezentsev',
            nconfig=1,
            ninputs=2,
            nparams=1,
            nstates=1,
            noutputs_max=2,
            nens_params=nens_params,
            nens_states=nens_states,
            nens_outputs=nens_outputs)

        self.config.names = 'dummy'
        self.config.units = '-'

        self._params.names = ['n']
        self._params.units = ['-']
        self._params.min = [0.5]
        self._params.max = [5.]
        self._params.default = [2.3]

        self.reset()


    def runblock(self, istart, iend, seed=None):
        kk = range(istart, iend+1)

        P = self.inputs[kk,0]
        PE = self.inputs[kk,1]
        n = self.params[0]
        Q = P*(1.-1./(1.+(P/PE)**n)**(1./n))
        E = P-Q
        self.outputs[kk, 0] = Q

        if self.outputs.shape[1] > 1:
            self.outputs[kk, 1] = E



class CalibrationTurcMezentsev(Calibration):

    def __init__(self, timeit=False):

        tm = TurcMezentsev()

        Calibration.__init__(self,
            model = tm, \
            ncalparams = 1, \
            timeit = timeit)

        self._calparams.means =  [2.3]
        self._calparams.stdevs = [1.]


    def cal2true(self, calparams):
        return calparams


