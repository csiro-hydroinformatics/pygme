
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

WAPABA_TMEAN = np.array([1.12, 0.57, -0.77, 5.65, -3.07])

WAPABA_TCOV = np.array([
    [0.091, -0.003, -0.016, -0.099, 0.083],
    [-0.003, 0.1, 0.01, -0.04, -0.03],
    [-0.016, 0.01, 0.56, 0.07, 0.17],
    [-0.099, -0.04, 0.07, 0.52, -0.05],
    [0.083, -0.03, 0.17, -0.05, 1.4]
])

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
        defaults =[
            3.21, #ALPHA1
            1.91, #ALPHA2
            0.56, #BETA
            445.81, #SMAX
            0.10, #INVK
        ]

        mins = [
            1.26, #ALPHA1
            1.01, #ALPHA2
            0.001, #BETA
            20., #SMAX
            0.001, #INVK
        ]

        maxs = [
            10.00, #ALPHA1
            10.00, #ALPHA2
            1.00, #BETA
            5000.00, #SMAX
            1.00, #INVK
        ]
        vect = Vector(["ALPHA1", "ALPHA2", "BETA", "SMAX", "INVK"], \
                defaults=defaults, \
                mins=mins, maxs=maxs)
        params = ParamsVector(vect)

        # State vector
        states = Vector(["S", "G"], check_bounds=False)

        # Model
        # 2 inputs : P, E
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

    def __init__(self, objfun=ObjFunBCSSE(0.5), \
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

        # Build parameter library from
        # MVT norm in transform space using latin hypercube
        tplib = sutils.lhs_norm(nparamslib, WAPABA_TMEAN, WAPABA_TCOV)

        # Back transform
        plib = tplib * 0.
        for i in range(len(plib)):
            plib[i, :] = wapaba_trans2true(tplib[i, :])
        plib = np.clip(plib, model.params.mins, model.params.maxs)
        self.paramslib = plib

