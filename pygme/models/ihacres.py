
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
def logit_fwd(u, eps=1e-7):
    u = u.clip(eps, 1-eps)
    return np.log(u/(1-u))

def logit_inv(v):
    return np.exp(v)/(1+np.exp(v))

def ihacres_trans2true(xt):
    x = logit_inv(xt)
    x[-1] = math.exp(xt[-1]) # d parameter
    return x

def ihacres_true2trans(x):
    xt = logit_fwd(x)
    xt[-1] = math.log(max(1e-2, x[-1])) # d parameter
    return xt


# Model
class IHACRES(Model):

    def __init__(self):

        # Config vector
        config = Vector(["shape"], [0], [0], [10])

        # params vector
        vect = Vector(["f", "e", "d"], \
                    [0.7, 0.16, 200], [0., 0., 1e-2], [1, 1, 2e3])
        params = ParamsVector(vect)

        # State vector
        states = Vector(["M"])

        # Model
        super(IHACRES, self).__init__("IHACRES",
            config, params, states, \
            ninputs=2, \
            noutputsmax=4)

        # Runoff outputs is named 'U' in ihacres code
        # we use 'Q' here to be compatible with other models
        self.outputs_names = ["Q", "M", "Mf", "ET"]


    def initialise_fromdata(self):
        """ Initialisation of IHACRES using
            * Production store: 50% filling level
            * Routing store: 30% filling level
        """
        d = self.params.d

        # Production store
        M0 = 0.5 * d

        # Model initialisation
        self.initialise(states=[M0])


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

    def __init__(self, objfun=ObjFunBCSSE(0.5), \
            warmup=36, \
            timeit=False,\
            fixed=None, \
            nparamslib=500, \
            objfun_kwargs={}):

        # Input objects for Calibration class
        model = IHACRES()
        params = model.params

        cp = Vector(["tf", "te", "td"], \
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

        # Sample parameter library from latin hyper-cube
        mean = params.defaults
        cov = np.diag([0.2, 0.2, 100])**2

        plib = sutils.lhs_norm(nparamslib, mean, cov)
        plib = np.clip(plib, params.mins, params.maxs)
        self.paramslib = plib


