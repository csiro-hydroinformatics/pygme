import math
import numpy as np
import pandas as pd

from scipy.stats import norm

from hydrodiy.data.containers import Vector
from hydrodiy.stat import sutils

from pygme.model import Model, ParamsVector, UH, ParameterCheckValueError
from pygme.calibration import Calibration, CalibParamsVector, ObjFunBCSSE

from pygme import has_c_module
if has_c_module("models_hydromodels"):
    import c_pygme_models_hydromodels

# Transformed parameters mean and covariance
GR4J_TMEAN = np.array([6., -0.8, 3., 0.7])
GR4J_TCOV = np.array([[1.16, 0.4, 0.15, -0.2],
        [0.4, 1.6, -0.3, -0.17],
        [0.15, -0.3, 1.68, -0.3],
        [-0.2, -0.17, -0.3, 0.6]])


def compute_PmEm(rain, evap):
    """ Compute rain and evap statistics needed for GR4J initialisation """
    # Check data
    rain = np.array(rain).astype(np.float64)
    evap = np.array(evap).astype(np.float64)
    PmEm = np.zeros(2, dtype=np.float64)

    # Run C code
    ierr = c_pygme_models_hydromodels.compute_PmEm(rain, evap, PmEm)

    if ierr > 0:
        raise ValueError(("c_pygme_models_hydromodels.compute_PmEm" +
            " returns {0}").format(ierr))

    return PmEm[0], PmEm[1]


def gr4j_X1_initial(Pm, Em, X1):
    """ Compute optimised filling level for the production store of GR4J """
    has_c_module("models_hydromodels")
    # Check data
    Pm = np.float64(Pm)
    Em = np.float64(Em)
    X1 = np.float64(X1)
    solution = np.zeros(1, dtype=np.float64)

    # Run C code
    ierr = c_pygme_models_hydromodels.gr4j_X1_initial(Pm, Em, X1, solution)

    if ierr > 0:
        raise ValueError(("c_pygme_models_hydromodels.gr4j_X1_initial" +
            " returns {0}").format(ierr))

    return solution[0]

# Transformation functions for gr4j parameters
def gr4j_trans2true(x):
    return np.array([ \
                math.exp(x[0]), \
                math.sinh(x[1]), \
                math.exp(x[2]), \
                0.49+math.exp(x[3])
            ])

def gr4j_true2trans(x):
    return np.array([ \
                math.log(max(1e-10, x[0])), \
                math.asinh(x[1]), \
                math.log(max(1e-10, x[2])), \
                math.log(max(1e-10, x[3]-0.49))
            ])
# Model
class GR4J(Model):

    def __init__(self, Pm=0., Em=0.):

        # Config vector - used to initialise model
        config = Vector(["nodata"], [0], [0], [1])

        # params vector
        vect = Vector(["X1", "X2", "X3", "X4"], \
                    defaults=[400, -1, 50, 0.5], \
                    mins=[1, -50, 1, 0.5], \
                    maxs=[1e4, 50, 1e4, 50])

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
        states = Vector(["S", "R"], check_bounds=False)

        # Model
        super(GR4J, self).__init__("GR4J",
            config, params, states, \
            ninputs=2, \
            noutputsmax=11)

        self.inputs_names = ["Rain", "PET"]
        self.outputs_names = ["Q", "S", "R", "ECH", "AE", \
                    "PR", "QD", "QR", "PERC", "Q1", "Q9"]


    def initialise_fromdata(self, Pm=0., Em=0.):
        """ Initialisation of GR4J using
            * Production store: steady state solution from Le Moine
              (2008, Page 212)
            * Routing store: 30% filling level

            Reference:
            Le Moine, Nicolas. "Le bassin versant de surface vu par
            le souterrain: une voie d"amelioration des performances
            et du realisme des modeles pluie-debit?."
            PhD diss., Paris 6, 2008.
        """
        X1 = self.params.X1
        X3 = self.params.X3

         # Production store
        if (Pm > 1e-10 or Em > 1e-10):
            # Using the routine from Le Moine
            S0 = X1 * gr4j_X1_initial(Pm, Em, X1)
        else:
            # Default GR4J initialisation
            S0 = 0.5 * X1

        # Routing store
        R0 = X3 * 0.3

        # Model initialisation
        self.initialise(states=[S0, R0])


    def run(self):
        # Get uh object (not set_timebase function, see ParamsVector class)
        has_c_module("models_hydromodels")

        _, uh1 = self.params.uhs[0]
        _, uh2 = self.params.uhs[1]

        # Run gr4j c code
        ierr = c_pygme_models_hydromodels.gr4j_run(uh1.nord, \
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
            raise ValueError(("c_pygme_models_hydromodels.gr4j_run" +
                " returns {0}").format(ierr))


class CalibrationGR4J(Calibration):

    def __init__(self, objfun=ObjFunBCSSE(0.5), \
                    warmup=5*365, \
                    timeit=False, \
                    fixed=None, \
                    objfun_kwargs={}, \
                    nparamslib=2000, \
                    Pm=0, Em=0):

        # Input objects for Calibration class
        model = GR4J()
        params = model.params
        # initialisation of states
        if Pm < 0 or Em < 0 :
            raise ValueError("Expected Pm and Em >0, "+\
                "got Pm={0}, Em={1}".format(Pm, Em))

        # Calib param vector
        cp = Vector(["tX1", "tX2", "tX3", "tX4"], \
                mins=gr4j_true2trans(params.mins),
                maxs=gr4j_true2trans(params.maxs),
                defaults=gr4j_true2trans(params.defaults))

        calparams = CalibParamsVector(model, cp, \
            trans2true=gr4j_trans2true, \
            true2trans=gr4j_true2trans,\
            fixed=fixed)

        # Instanciate calibration
        super(CalibrationGR4J, self).__init__(calparams, \
            objfun=objfun, \
            warmup=warmup, \
            timeit=timeit, \
            objfun_kwargs=objfun_kwargs,
            initial_kwargs={"Pm":Pm, "Em":Em})

        # Build parameter library from
        # MVT norm in transform space using latin hypercube
        tplib = sutils.lhs_norm(nparamslib, GR4J_TMEAN, GR4J_TCOV)

        # Back transform
        plib = tplib * 0.
        for i in range(len(plib)):
            plib[i, :] = gr4j_trans2true(tplib[i, :])
        plib = np.clip(plib, model.params.mins, model.params.maxs)
        self.paramslib = plib


