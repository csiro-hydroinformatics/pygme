
import math
import numpy as np
import pandas as pd

from hydrodiy.data.containers import Vector

from hydrodiy.stat import sutils

from pygme.model import Model, ParamsVector
from pygme.calibration import Calibration, CalibParamsVector, ObjFunBCSSE

import c_pynonstat

# Transformation functions for gr4j parameters
def wapaba_trans2true(x):
    return np.exp(x)

def wapaba_true2trans(x):
    return np.log(np.maximum(1e-10, x))

# Model
class WAPABA(Model):

    def __init__(self, version):
        # Config vector
        version = int(version)
        config = Vector(["version"], \
                            [version], \
                            [version-1e-10], \
                            [version+1e-10])

        # params vector
        if version == 0:
            vect = Vector(["ALPHA1", "ALPHA2", "BETA", "SMAX", "INVK"], \
                    defaults=[2., 2., 0.5, 100., 0.5], \
                    mins=[1.01, 1.01, 1e-3, 1., 1e-3], \
                    maxs=[10., 10., 1., 5000., 1.0])
        else:
            raise ValueError(f"Expected version to be 0, got {version}.")

        params = ParamsVector(vect)

        # State vector
        states = Vector(["S", "G"])

        # Model
        # 3 inputs : P, E and days in month
        super(WAPABA, self).__init__("WAPABA",
            config, params, states, \
            ninputs=3, \
            noutputsmax=10)

        self.outputs_names = ["Q", "S", "G", "ET", "F1", "F2", "R", \
                            "Qb", "Qs", "W"]

    @property
    def version(self):
        return np.int32(self.config.version)


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
        ierr = c_pynonstat.wapaba_run(self.istart, self.iend,
            self.config.values, \
            self.params.values, \
            self.inputs, \
            self.states.values, \
            self.outputs)

        if ierr > 0:
            raise ValueError("Model wapaba,"+\
                    f" c_pynonstat.wapaba_run returns {ierr}")



class CalibrationWAPABA(Calibration):

    def __init__(self, version, objfun=ObjFunBCSSE(0.2), \
            warmup=36, \
            timeit=False,\
            fixed=None, \
            nparamslib=500, \
            objfun_kwargs={}):

        # Input objects for Calibration class
        model = WAPABA(version)
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
        cov = np.diag((params.maxs-params.mins)/2)
        plib = sutils.lhs_norm(nparamslib, mean, cov)
        plib = np.clip(plib, params.mins, params.maxs)
        self.paramslib = plib


