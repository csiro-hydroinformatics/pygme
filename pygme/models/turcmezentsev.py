
import math
import numpy as np
import pandas as pd
from scipy.special import lambertw

from hydrodiy.data.containers import Vector

from pygme.model import Model, ParamsVector
from pygme.calibration import Calibration, CalibParamsVector, ObjFunBCSSE


def turcm_forward(rain, evap, nparam):
    """ Turc-Mezentsev forward model """
    return rain*(1.-1./np.power(1.+np.power(rain/evap, nparam), (1./nparam)))


def turcm_iterfun(x, rc, ar):
    """ Function used in Halley"s iteration """
    ax = np.power(ar, x)
    return x*np.log(1-rc)+np.log(1+ax), \
            np.log(1-rc)+np.log(ar)*ax/(1+ax), \
            np.log(ar)**2*ax**2/(1+ax)**2


def turcm_backward(runoff, rain, evap, nitermax=20, tol=1e-3):
    """ Turc-Mezentsev inverse model.
        Compute the parameter n from runoff, rain and evap data
    """
    # Dimensionless variables
    rc = runoff/rain
    ar = rain/evap

    if np.any(ar > 2.):
        raise ValueError("Expected aridity index < 2, got "+\
            "{:0.2f}".format(np.max(ar)))

    # First approximation
    nparam0 = -lambertw(np.log(ar)/np.log(1-rc))/np.log(ar)
    nparam0 = nparam0.real

    # Hayley iteration
    # See https://en.wikipedia.org/wiki/Halley%27s_method

    # initialise
    niter = 0
    err = 1e10
    errprev = 1e10
    if hasattr(nparam0, "data"):
        nparam = nparam0.copy()
    else:
        nparam = nparam0

    f0, f1, f2 = turcm_iterfun(nparam0, rc, ar)
    err0 = np.max(np.abs(f0))

    # Loop
    while niter < nitermax and err > tol:
        f0, f1, f2 = turcm_iterfun(nparam, rc, ar)
        nparam += -2*f0*f1/(2*f1**2-f0*f2)
        err = np.max(np.abs(f0))

        # Monitor convergence
        if err > errprev*5:
            break

        niter += 1
        errprev = err

    # Check convergence improves
    if err > err0:
        nparam = nparam0

    return nparam, niter


class TurcMezentsev(Model):

    def __init__(self):

        # Config vector
        config = Vector(["continuous"],\
                    [0], [0], [1])

        # params vector
        vect = Vector(["n"], [2.3],  [0.5], [5])
        params = ParamsVector(vect)

        # State vector
        states = Vector(["S"])

        # Model
        super(TurcMezentsev, self).__init__("TurcMezentsev",
            config, params, states, \
            ninputs=2, \
            noutputsmax=2)

        self.inputs_names = ["Rain", "PET"]
        self.outputs_names = ["Q", "E"]


    def run(self):
        istart, iend = self.istart, self.iend
        kk = range(istart, iend+1)

        P = self.inputs[kk,0]
        PE = self.inputs[kk,1]
        n = self.params.values[0]
        Q = turcm_forward(P, PE, n)
        E = P-Q
        self.outputs[kk, 0] = Q

        if self.outputs.shape[1] > 1:
            self.outputs[kk, 1] = E



class CalibrationTurcMezentsev(Calibration):

    def __init__(self, objfun=ObjFunBCSSE(0.5), \
                    warmup=2, \
                    objfun_kwargs={}, \
                    nparamslib=500):

        # Input objects for Calibration class
        model = TurcMezentsev()
        params = model.params
        calparams = CalibParamsVector(model)

        # Instanciate calibration
        super(CalibrationTurcMezentsev, self).__init__(calparams, \
            objfun=objfun, \
            warmup=warmup, \
            objfun_kwargs=objfun_kwargs)

        # Parameter library
        plib = np.random.multivariate_normal(mean=params.defaults, \
                    cov=np.diag((params.maxs-params.mins)/3), \
                    size=nparamslib)
        plib = np.clip(plib, params.mins, params.maxs)
        self.paramslib = plib

