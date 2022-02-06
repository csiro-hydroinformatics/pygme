import math
import numpy as np
import pandas as pd

from hydrodiy.stat import sutils
from hydrodiy.data.containers import Vector

from pygme.model import Model, ParamsVector, UH, ParameterCheckValueError
from pygme.calibration import Calibration, CalibParamsVector, ObjFunBiasBCSSE
from pygme.models.gr4j import gr4j_X1_initial

from pygme import has_c_module
if has_c_module("models_hydromodels"):
    import c_pygme_models_hydromodels

# Mean and covariance of transformed parameters
GR6J_TMEAN = np.array([5.07, -0.20, 2.58, 0.21, 0.07, 1.21])
GR6J_TCOV = np.array([
        [2.5, 0.21, -1.1, 0.06, 0.3, -1.1],
    	[0.2, 0.25, -0.2, -0.04, 0.0, -0.1],
    	[-1.1, -0.22, 2.1, -0.22, -0.1, 1.5],
    	[0.1, -0.04, -0.2, 0.48, 0.0, -0.2],
    	[0.3, 0.02, -0.1, 0.03, 2.1, -0.0],
    	[-1.1, -0.11, 1.5, -0.22, -0.0, 4.9],
    ])

# Transformation functions for gr6j parameters
def gr6j_trans2true(x):
    return np.array([ \
                math.exp(x[0]), \
                math.sinh(x[1]), \
                math.exp(x[2]), \
                0.49+math.exp(x[3]), \
                math.sinh(x[4]), \
                math.exp(x[5])
            ])

def gr6j_true2trans(x):
    return np.array([ \
                math.log(max(1e-10, x[0])), \
                math.asinh(x[1]), \
                math.log(max(1e-10, x[2])), \
                math.log(max(1e-10, x[3]-0.49)), \
                math.asinh(x[4]), \
                math.log(max(1e-10, x[5]))
            ])
# Model
class GR6J(Model):

    def __init__(self):

        # Config vector
        config = Vector(["version"], [0])

        # params vector
        defaults = [
        		353.38, #X1
        		-0.26, #X2
        		47.71, #X3
        		2.12, #X4
        		0.03, #X5
        		48.94, #X6
        ]

        mins = [
        		1.00, #X1
        		-21.30, #X2
        		1.00, #X3
        		0.50, #X4
        		-50.00, #X5
        		0.10, #X6
        ]

        maxs = [
        		10000.00, #X1
        		6.67, #X2
        		10000.00, #X3
        		50.00, #X4
        		50.00, #X5
        		1000.00, #X6
        ]

        vect = Vector(["X1", "X2", "X3", "X4", "X5", "X6"], \
                    defaults=defaults, \
                    mins=mins, maxs=maxs)

        # Rule to exclude certain parameters
        def checkvalues(values):
            X2, X3 = values[1:3]
            if X3 < -X2:
                raise ParameterCheckValueError(\
                        "X3 ({0:0.2f}) < - X2 ({1:0.2f})".format(X3, X2))

        params = ParamsVector(vect, checkvalues=checkvalues)

        # UH
        params.add_uh("gr4j_ss1_daily", lambda params: params.X4)
        params.add_uh("gr4j_ss2_daily", lambda params: params.X4)

        # State vector
        # Set default state vector to -100 for exponential reservoir
        states = Vector(["S", "R", "A"], [0., 0., -100], \
                                check_bounds=False)

        # Model
        super(GR6J, self).__init__("GR6J",
            config, params, states, \
            ninputs=2, \
            noutputsmax=13)

        self.outputs_names = ["Q", "S", "R", "A", "ECH", "AE", \
                    "PR", "QD", "QR", "PERC", "QExp", "Q1", "Q9"]


    def initialise_fromdata(self, Pm=0., Em=0., Q0=1e-3):
        """ Initialisation of GR6J using
            * Production store: steady state solution from Le Moine
              (2008, Page 212)
            * Routing store: 30% filling level
            * Exponential store: using Equation 8 in Michel et al. (2003)

            Reference:
            Le Moine, Nicolas. "Le bassin versant de surface vu par
            le souterrain: une voie d"amelioration des performances et du
            realisme des modeles pluie-debit?." PhD diss., Paris 6, 2008.

            Michel, Claude, Charles Perrin, and Vazken Andréassian.
            "The exponential store: a correct formulation for
            rainfall—runoff modelling." Hydrological Sciences
            Journal 48.1 (2003): 109-124.
        """
        X1 = self.params.values[0]
        X3 = self.params.values[2]
        X6 = self.params.values[5]

        # Production store
        if Pm > 1e-10 or Em > 1e-10:
            # Using the routine from Le Moine
            S0 = X1 * gr4j_X1_initial(Pm, Em, X1)
        else:
            S0 = 0.5 * X1

        # Routing store
        R0 = 0.3 * X3

        # Exponential store
        A0 = X6*math.log(math.exp(Q0/X6)-1)

        # Model initialisation
        self.initialise(states=[S0, R0, A0])



    def run(self):
        has_c_module("models_hydromodels")

        # Get version
        version = np.int32(self.config.version)

        # Get uh object (not set_timebase function, see ParamsVector class)
        _, uh1 = self.params.uhs[0]
        _, uh2 = self.params.uhs[1]

        # Run gr4j c code
        ierr = c_pygme_models_hydromodels.gr6j_run(\
            version, \
            uh1.nord, \
            uh2.nord, self.istart, self.iend, \
            self.params.values, \
            uh1.ord, \
            uh2.ord, \
            self.inputs, \
            uh1.states, \
            uh2.states, \
            self.states.values, \
            self.outputs)

        if ierr > 0:
            raise ValueError(("c_pygme_models_hydromodels.gr6j_run" +
                " returns {0}").format(ierr))


class CalibrationGR6J(Calibration):

    def __init__(self, objfun=ObjFunBiasBCSSE(0.5), \
                    warmup=5*365, \
                    timeit=False, \
                    fixed=None, \
                    nparamslib=8000, \
                    objfun_kwargs={},
                    Pm=0., Em=0.):

        # Input objects for Calibration class
        model = GR6J()
        params = model.params

        # initialisation of states
        if Pm < 0 or Em < 0 :
            raise ValueError("Expected Pm and Em >0, "+\
                "got Pm={0}, Em={1}".format(Pm, Em))

        initial_kwargs = {"Pm":Pm, "Em":Em}

        # Build calib vector
        cp = Vector(["tX1", "tX2", "tX3", "tX4", "tX5", "tX6"], \
                mins=gr6j_true2trans(params.mins),
                maxs=gr6j_true2trans(params.maxs),
                defaults=gr6j_true2trans(params.defaults))

        calparams = CalibParamsVector(model, cp, \
            trans2true=gr6j_trans2true, \
            true2trans=gr6j_true2trans,\
            fixed=fixed)

        # Instanciate calibration
        super(CalibrationGR6J, self).__init__(calparams, \
            objfun=objfun, \
            warmup=warmup, \
            timeit=timeit, \
            objfun_kwargs=objfun_kwargs, \
            initial_kwargs=initial_kwargs)

        # Build parameter library from
        # MVT norm in transform space
        tplib = sutils.lhs_norm(nparamslib, GR6J_TMEAN, GR6J_TCOV)

        # Back transform parameter library
        plib = tplib * 0.
        for i in range(len(plib)):
            plib[i, :] = gr6j_trans2true(tplib[i, :])

        plib = np.clip(plib, model.params.mins, model.params.maxs)
        self.paramslib = plib

