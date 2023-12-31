import math, re
import numpy as np
import pandas as pd

from hydrodiy.data.containers import Vector

from hydrodiy.stat import sutils

from pygme.model import Model, ParamsVector
from pygme.calibration import Calibration, CalibParamsVector, ObjFunBCSSE

from pygme import has_c_module
if has_c_module("models_hydromodels"):
    import c_pygme_models_hydromodels

IHACRES_TMEAN = np.array([-0.73, 6.47, -2.56])
IHACRES_TCOV = np.array([\
        [0.61, -0.11, -0.13],
        [-0.11, 0.73, -0.07],
        [-0.13, -0.07, 2.2]
    ])


def get_shapefactor(name):
    return float(re.sub("^IHACRES", "", name))

# Transformation functions for ihacres parameters
def logit_fwd(u, eps=1e-7):
    u = u.clip(eps, 1-eps)
    return np.log(u/(1-u))

def logit_inv(v):
    return np.exp(v)/(1+np.exp(v))

def ihacres_trans2true(xt):
    x = np.zeros(3)
    x[0] = 2*logit_inv(xt[0]) # f parameter
    x[1] = math.exp(xt[1]) # d parameter
    x[2] = math.exp(xt[2]) # delta parameter
    return x

def ihacres_true2trans(x):
    xt = np.zeros(3)
    xt[0] = logit_fwd(x[0]/2) # f parameter
    xt[1] = math.log(max(1e-2, x[1])) # d parameter
    xt[2] = math.log(max(1e-2, x[2])) # delta parameter
    return xt


# Model
class IHACRES(Model):

    def __init__(self, shapefactor=0):
        # Config vector
        # e is normally an IHACRES parameter used if evap inputs are
        # different from PET or if vegetation plays a big role in
        # influencing evap.
        config = Vector(["shapefactor", "e"], [shapefactor, 1], [0, 0.1], [10, 1.5])

        # Parameter
        defaults = [
                0.64, #f
                826.05, #d
                0.26 # delta
        ]

        mins = [
                0.10, #f
                28.05, #d
                0.01 # delta
        ]

        maxs = [
                2.00, #f
                3000.00, #d
                24 # delta
        ]

        # params vector
        vect = Vector(["f", "d", "delta"], \
                        defaults=defaults, \
                        mins=mins, \
                        maxs=maxs)
        params = ParamsVector(vect)

        # State vector
        states = Vector(["M", "R"])

        # Model
        super(IHACRES, self).__init__("IHACRES",
            config, params, states, \
            ninputs=2, \
            noutputsmax=12)

        self.inputs_names = ["Rain", "PET"]

        # Runoff outputs is named 'U' in ihacres code
        # we use 'Q' here to be compatible with other models
        self.outputs_names = ["Q", "M", "Mf", "ET", \
                                "U0", "F", "M0", "M1", "L0", "L1", "U", "R"]


    def initialise_fromdata(self):
        """ Initialisation of IHACRES using
            * Production store: 50% filling level
            * Routing store: 30% filling level
        """
        # Production store
        M0 = 0.5 * self.params.d

        # Routing store
        R0 = 100. * self.params.delta

        # Model initialisation
        self.initialise(states=[M0, R0])


    def run(self):
        has_c_module("models_hydromodels")

        ierr = c_pygme_models_hydromodels.ihacres_run(self.istart, self.iend,
            self.config.values, \
            self.params.values, \
            self.inputs, \
            self.states.values, \
            self.outputs)

        if ierr > 0:
            raise ValueError(("Model ihacres, c_pygme_models_hydromodels.ihacres_run" + \
                    "returns {0}").format(ierr))



class CalibrationIHACRES(Calibration):

    def __init__(self, shapefactor=0, objfun=ObjFunBCSSE(0.5), \
            warmup=36, \
            timeit=False,\
            fixed=None, \
            nparamslib=500, \
            objfun_kwargs={}):

        # Input objects for Calibration class
        model = IHACRES(shapefactor)
        params = model.params

        cp = Vector(["tf", "td", "tdelta"], \
                mins=ihacres_true2trans(params.mins),
                maxs=ihacres_true2trans(params.maxs),
                defaults=ihacres_true2trans(params.defaults))
        calparams = CalibParamsVector(model, cp, \
            trans2true=ihacres_trans2true, \
            true2trans=ihacres_true2trans, \
            fixed=fixed)

        # Initialisation arguments
        initial_kwargs = {}

        # Instanciate calibration
        super(CalibrationIHACRES, self).__init__(calparams, \
            objfun=objfun, \
            warmup=warmup, \
            timeit=timeit, \
            objfun_kwargs=objfun_kwargs, \
            initial_kwargs=initial_kwargs)

        # Build parameter library from
        # MVT norm in transform space using latin hypercube
        tplib = sutils.lhs_norm(nparamslib, IHACRES_TMEAN, IHACRES_TCOV)

        # Back transform
        plib = tplib * 0.
        for i in range(len(plib)):
            plib[i, :] = ihacres_trans2true(tplib[i, :])
        plib = np.clip(plib, model.params.mins, model.params.maxs)
        self.paramslib = plib

