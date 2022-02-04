
import math
import numpy as np
import pandas as pd

from hydrodiy.data.containers import Vector

from hydrodiy.stat import sutils

from pygme.model import Model, ParamsVector
from pygme.calibration import Calibration, CalibParamsVector, ObjFunBiasBCSSE

from pygme import has_c_module
if has_c_module("models_hydromodels"):
    import c_pygme_models_hydromodels

WAPABA_TMEAN = np.log(np.array([2., 2., 0.5, 100, 0.5]))
WAPABA_TCOV = np.eye(5)

# Transformation functions for gr4j parameters
def wapaba_trans2true(x):
    return np.exp(x)

def wapaba_true2trans(x):
    return np.log(np.maximum(1e-10, x))

# Model
class WAPABA(Model):

    def __init__(self):
        # Config vector
        config = Vector(["nodata"], [0], [0], [1])

        # params vector
        vect = Vector(["ALPHA1", "ALPHA2", "BETA", "SMAX", "INVK"], \
                defaults=[2., 2., 0.5, 100., 0.5], \
                mins=[1.01, 1.01, 1e-3, 1., 1e-3], \
                maxs=[10., 10., 1., 5000., 1.0])
        params = ParamsVector(vect)

        # State vector
        states = Vector(["S", "G"])

        # Model
        # 3 inputs : P, E and days in month
        super(WAPABA, self).__init__("WAPABA",
            config, params, states, \
            ninputs=2, \
            noutputsmax=10)

        self.outputs_names = ["Q", "S", "G", "ET", "F1", "F2", "R", \
                            "Qb", "Qs", "W"]


    def initialise_fromdata(self):
        """ Initialisation of GR2M using
            * Production store: 50% filling level
            * Groundwater store: 0% filling level
        """
        # Production store
        SMAX = self.params.SMAX
        S0 = 0.5 * SMAX

        # Groundwater store
        G0 = 0.

        # Model initialisation
        self.initialise(states=[S0, G0])


    def run(self):
        ierr = c_pygme_models_hydromodels.wapaba_run(self.istart, self.iend,
                        self.params.values, \
                        self.inputs, \
                        self.states.values, \
                        self.outputs)

        errmsg = "Model wapaba,"+\
                    f" c_pygme_models_hydromodels.wapaba_run returns {ierr}"
        assert ierr==0, errmsg



class CalibrationWAPABA(Calibration):

    def __init__(self, objfun=ObjFunBiasBCSSE(0.5), \
            warmup=36, \
            timeit=False,\
            fixed=None, \
            nparamslib=5000, \
            objfun_kwargs={}):

        # Input objects for Calibration class
        model = WAPABA()
        params = model.params

        pnames =["tALPHA1", "tALPHA2", "tBETA", "tSMAX", "tINVK"]

        cp = Vector(pnames, \
                mins=wapaba_true2trans(params.mins),
                maxs=wapaba_true2trans(params.maxs),
                defaults=wapaba_true2trans(params.defaults))

        calparams = CalibParamsVector(model, cp, \
            trans2true=wapaba_trans2true, \
            true2trans=wapaba_true2trans, \
            fixed=fixed)

        # Initialisation arguments
        initial_kwargs = {}

        # Instanciate calibration
        super(CalibrationWAPABA, self).__init__(calparams, \
            objfun=objfun, \
            warmup=warmup, \
            timeit=timeit, \
            objfun_kwargs=objfun_kwargs, \
            initial_kwargs=initial_kwargs)

        # Sample parameter library from latin hyper-cube
        mean = params.defaults
        plib = sutils.lhs_norm(nparamslib, WAPABA_TMEAN, WAPABA_TCOV)
        plib = np.clip(plib, params.mins, params.maxs)
        self.paramslib = plib


