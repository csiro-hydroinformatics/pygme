
import math
import numpy as np
import pandas as pd

from hydrodiy.data.containers import Vector

from pygme.model import Model, ParamsVector
from pygme.calibration import Calibration, CalibParamsVector


class TurcMezentsev(Model):

    def __init__(self):

        # Config vector
        config = Vector(['continuous'],\
                    [0], [0], [1])

        # params vector
        vect = Vector(['n'], [2.3],  [0.5], [5])
        params = ParamsVector(vect)

        # State vector
        states = Vector(['S'])

        # Model
        super(TurcMezentsev, self).__init__('TurcMezentsev',
            config, params, states, \
            ninputs=2, \
            noutputsmax=2)


    def run(self):
        istart, iend = self.istart, self.iend
        kk = range(istart, iend+1)

        P = self.inputs[kk,0]
        PE = self.inputs[kk,1]
        n = self.params.values[0]
        Q = P*(1.-1./np.power(1.+np.power(P/PE, n), (1./n)))
        E = P-Q
        self.outputs[kk, 0] = Q

        if self.outputs.shape[1] > 1:
            self.outputs[kk, 1] = E



class CalibrationTurcMezentsev(Calibration):

    def __init__(self, timeit=False):

        # Input objects for Calibration class
        model = TurcMezentsev()
        params = model.params

        plib = np.random.multivariate_normal(mean=params.defaults, \
                    cov=np.diag((params.maxs-params.mins)/3), \
                    size=500)
        plib = np.clip(plib, params.mins, params.maxs)

        calparams = CalibParamsVector(model)

        # Instanciate calibration
        super(CalibrationTurcMezentsev, self).__init__(calparams, \
            paramslib=plib)


