import math
import numpy as np
import pandas as pd

from hydrodiy.data.containers import Vector
from pygme.model import Model, ParamsVector
from pygme.calibration import Calibration, CalibParamsVector, ObjFunBCSSE

from pygme import has_c_module
if has_c_module("models_hydromodels"):
    import c_pygme_models_hydromodels
    import c_pygme_models_utils

# Transformation functions for lagroute parameters
def lagroute_trans2true(x):
    return x

def lagroute_true2trans(x):
    return x


class LagRoute(Model):

    def __init__(self):

        # Config vector
        config = Vector(["timestep", "length", \
                            "flowref", "storage_expon"], \
                    defaults=[86400, 1e5, 1, 1], \
                    mins=[1, 1, 1e-6, 1], \
                    maxs=[np.inf, np.inf, np.inf, 2])

        # params vector
        vect = Vector(["U", "alpha"], \
                    defaults=[1., 0.5], \
                    mins=[0.01, 0.], \
                    maxs=[20., 1.])
        params = ParamsVector(vect)

        # Attach config to params to
        # retrieve data during UH computation
        params.config = config

        # Uh calculation
        def compute_delta(params):
            # Get config data
            config = params.config

            # Compute Lag = alpha * U * L / dt
            delta = config.length * params.U * params.alpha
            delta /= config.timestep
            delta = np.float64(delta)

            if np.isnan(delta):
                raise ValueError("Expected non nan value for delta. "+\
                    "length={0}, timestep={1}, U={2}, alpha={3}".format(\
                        config.length, config.timestep, params.U, \
                        params.alpha))

            return delta

        params.add_uh("lag", compute_delta)

        # State vector
        states = Vector(["S"])

        # Model
        super(LagRoute, self).__init__("LagRoute",
            config, params, states, \
            ninputs=1, \
            noutputsmax=4)

        self.outputs_names = ["Q", "Q1lag", \
                    "VR", "V1"]


    def run(self):
        has_c_module("models_hydromodels")

        # Get uh object (not set_timebase function, see ParamsVector class)
        _, uh = self.params.uhs[0]

        ierr = c_pygme_models_hydromodels.lagroute_run(uh.nord, \
            self.istart, self.iend,
            self.config.values, \
            self.params.values, \
            uh.ord, \
            self.inputs, \
            uh.states, \
            self.states.values, \
            self.outputs)

        if ierr > 0:
            raise ValueError(("c_pygme_models_hydromodels." + \
                "lagroute_run returns {0}").format(ierr))


class CalibrationLagRoute(Calibration):

    def __init__(self, objfun=ObjFunBCSSE(0.2), \
                    warmup=5*365, \
                    timeit=False, \
                    fixed=None, \
                    nparamslib=400, \
                    objfun_kwargs={}):

        # Input objects for Calibration class
        model = LagRoute()
        params = model.params

        cp = Vector(["tU", "talpha"], \
                mins=params.mins,
                maxs=params.maxs,
                defaults=params.defaults)

        # no parameter transformation
        calparams = CalibParamsVector(model, cp, \
            trans2true=lagroute_trans2true,\
            true2trans=lagroute_true2trans,\
            fixed=fixed)

        # Instanciate calibration
        super(CalibrationLagRoute, self).__init__(calparams, \
            objfun=objfun, \
            warmup=warmup, \
            timeit=timeit, \
            objfun_kwargs=objfun_kwargs)

        # Build parameter library from
        # systematic exploration of parameter space
        nn = int(math.sqrt(nparamslib))
        uu , aa = np.meshgrid(np.linspace(0.1, 3, nn), \
                        np.linspace(0, 1, nn))
        plib = np.column_stack([uu.ravel(), aa.ravel()])

        self.paramslib = plib

