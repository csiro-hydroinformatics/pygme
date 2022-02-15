
import math
import numpy as np
import pandas as pd

from hydrodiy.data.containers import Vector

from hydrodiy.stat import sutils

from pygme.model import Model, ParamsVector
from pygme.calibration import Calibration, CalibParamsVector, ObjFunBCSSE

from pygme import has_c_module
if has_c_module("models_hydromodels"):
    import c_pygme_models_hydromodels

# Transformation functions for gr4j parameters
def gr2m_trans2true(x):
    return np.exp(x)

def gr2m_true2trans(x):
    return np.log(np.maximum(1e-10, x))

# Model
class GR2M(Model):

    def __init__(self):

        # Config vector
        config = Vector(["catcharea"], [0], [0])

        # params vector
        vect = Vector(["X1", "X2"], \
                    [400, 0.8], [10., 0.1], [1e4, 3])
        params = ParamsVector(vect)

        # State vector
        states = Vector(["S", "R"])

        # Model
        super(GR2M, self).__init__("GR2M",
            config, params, states, \
            ninputs=2, \
            noutputsmax=10)

        self.outputs_names = ["Q", "S", "R", "F", "P1", "P2", "P3", \
                            "R1", "R2", "AE"]


    def initialise_fromdata(self):
        """ Initialisation of GR2M using
            * Production store: 50% filling level
            * Routing store: 30% filling level
        """
        X1 = self.params.X1

        # Production store
        S0 = 0.5 * X1

        # Routing store 60x30% = 18 mm
        R0 = 18

        # Model initialisation
        self.initialise(states=[S0, R0])


    def run(self):
        has_c_module("models_hydromodels")

        ierr = c_pygme_models_hydromodels.gr2m_run(self.istart, self.iend,
            self.params.values, \
            self.inputs, \
            self.states.values, \
            self.outputs)

        if ierr > 0:
            raise ValueError(("Model gr2m, c_pygme_models_hydromodels.gr2m_run" + \
                    "returns {0}").format(ierr))



class CalibrationGR2M(Calibration):

    def __init__(self, objfun=ObjFunBCSSE(0.5), \
            warmup=36, \
            timeit=False,\
            fixed=None, \
            nparamslib=500, \
            objfun_kwargs={}):

        # Input objects for Calibration class
        model = GR2M()
        params = model.params

        cp = Vector(["tX1", "tX2"], \
                mins=gr2m_true2trans(params.mins),
                maxs=gr2m_true2trans(params.maxs),
                defaults=gr2m_true2trans(params.defaults))
        calparams = CalibParamsVector(model, cp, \
            trans2true=gr2m_trans2true, \
            true2trans=gr2m_true2trans, \
            fixed=fixed)

        # Initialisation arguments
        initial_kwargs = {}

        # Instanciate calibration
        super(CalibrationGR2M, self).__init__(calparams, \
            objfun=objfun, \
            warmup=warmup, \
            timeit=timeit, \
            objfun_kwargs=objfun_kwargs, \
            initial_kwargs=initial_kwargs)

        # Sample parameter library from latin hyper-cube
        mean = params.defaults
        cov = np.diag((params.maxs-params.mins)/2)
        plib = sutils.lhs_norm(nparamslib, mean, cov)
        plib = np.clip(plib, params.mins, params.maxs)
        self.paramslib = plib


