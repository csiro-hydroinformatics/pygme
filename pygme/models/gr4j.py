import math
import numpy as np
import pandas as pd

from scipy.stats import norm

from hydrodiy.data.containers import Vector
from hydrodiy.stat import sutils

from pygme.model import Model, ParamsVector, UH
from pygme.calibration import Calibration, CalibParamsVector, ObjFunBCSSE

import c_pygme_models_hydromodels

def compute_PmEm(rain, evap):
    ''' Compute rain and evap statistics needed for GR4J initialisation '''
    # Check data
    rain = np.array(rain).astype(np.float64)
    evap = np.array(evap).astype(np.float64)
    PmEm = np.zeros(2, dtype=np.float64)

    # Run C code
    ierr = c_pygme_models_hydromodels.compute_PmEm(rain, evap, PmEm)

    if ierr > 0:
        raise ValueError(('c_pygme_models_hydromodels.compute_PmEm' +
            ' returns {0}').format(ierr))

    return PmEm[0], PmEm[1]


def lhs_params(tmean, tcov, nsamples):
    ''' Perform latin hypercube sampling of parameters in transformed space '''
    nvars = len(tmean)
    q = sutils.lhs(nsamples, [0]*nvars, [1]*nvars)
    nsmp = norm.ppf(q)
    S = np.linalg.cholesky(tcov)
    smp = tmean[:, None] + np.dot(S, nsmp.T)
    return smp.T


def gr4j_X1_initial(Pm, Em, X1):
    ''' Compute optimised filling level for the production store of GR4J '''
    # Check data
    Pm = np.float64(Pm)
    Em = np.float64(Em)
    X1 = np.float64(X1)
    solution = np.zeros(1, dtype=np.float64)

    # Run C code
    ierr = c_pygme_models_hydromodels.gr4j_X1_initial(Pm, Em, X1, solution)

    if ierr > 0:
        raise ValueError(('c_pygme_models_hydromodels.gr4j_X1_initial' +
            ' returns {0}').format(ierr))

    return solution[0]


class GR4J(Model):

    def __init__(self, Pm=0., Em=0.):

        # Config vector - used to initialise model
        config = Vector(['nodata'], [0], [0], [1])

        # params vector
        vect = Vector(['X1', 'X2', 'X3', 'X4'], \
                    [400, -1, 50, 0.5], \
                    [1, -50, 1, 0.5], \
                    [1e4, 50, 1e4, 50])
        params = ParamsVector(vect)

        # UH
        params.add_uh('gr4j_ss1_daily', lambda params: params.X4)
        params.add_uh('gr4j_ss2_daily', lambda params: params.X4)

        # State vector
        states = Vector(['S', 'R'], check_bounds=False)

        # Model
        super(GR4J, self).__init__('GR4J',
            config, params, states, \
            ninputs=2, \
            noutputsmax=9)

        self.outputs_names = ['Q', 'ECH', 'AE', \
                    'PR', 'QD', 'QR', 'PERC', 'S', 'R']


    def initialise_fromdata(self, Pm=0., Em=0.):
        ''' Initialisation of GR4J using
            * Production store: steady state solution from Le Moine
              (2008, Page 212)
            * Routing store: 30% filling level

            Reference:
            Le Moine, Nicolas. "Le bassin versant de surface vu par le souterrain: une voie
            d'amélioration des performances et du réalisme des modèles pluie-débit?." PhD diss., Paris 6, 2008.
        '''
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
            raise ValueError(('c_pygme_models_hydromodels.gr4j_run' +
                ' returns {0}').format(ierr))


class CalibrationGR4J(Calibration):

    def __init__(self, objfun=ObjFunBCSSE(0.2), \
                    warmup=5*365, \
                    timeit=False, \
                    fixed=None, \
                    objfun_kwargs={}, \
                    Pm=0, Em=0):

        # Input objects for Calibration class
        model = GR4J()
        params = model.params

        # Transformation functions for parameters
        trans2true = lambda x: np.array([
                        math.exp(x[0]), \
                        math.sinh(x[1]), \
                        math.exp(x[2]), \
                        0.49+math.exp(x[3])
                    ])

        true2trans = lambda x: np.array([
                        math.log(x[0]), \
                        math.asinh(x[1]), \
                        math.log(x[2]), \
                        math.log(x[3]-0.49)
                    ])

        # initialisation of states
        if Pm < 0 or Em < 0 :
            raise ValueError('Expected Pm and Em >0, '+\
                'got Pm={0}, Em={1}'.format(Pm, Em))

        # Calib param vector
        cp = Vector(['tX1', 'tX2', 'tX3', 'tX4'], \
                mins=true2trans(params.mins),
                maxs=true2trans(params.maxs),
                defaults=true2trans(params.defaults))

        calparams = CalibParamsVector(model, cp, \
            trans2true=trans2true, \
            true2trans=true2trans,\
            fixed=fixed)

        # Build parameter library from
        # MVT norm in transform space using latin hypercube
        tmean = np.array([6., -0.8, 3., 0.7])
        tcov = np.array([[1.16, 0.4, 0.15, -0.2],
                [0.4, 1.6, -0.3, -0.17],
                [0.15, -0.3, 1.68, -0.3],
                [-0.2, -0.17, -0.3, 0.6]])
        tplib = lhs_params(tmean, tcov, 2000)

        # Back transform
        plib = tplib * 0.
        plib[:, [0, 2, 3]] = np.exp(tplib[:, [0, 2, 3]])
        plib[:, 3] += 0.49
        plib[:, 1] = np.sinh(tplib[:, 1])
        plib = np.clip(plib, model.params.mins, model.params.maxs)

        # Instanciate calibration
        super(CalibrationGR4J, self).__init__(calparams, \
            objfun=objfun, \
            warmup=warmup, \
            timeit=timeit, \
            paramslib=plib, \
            objfun_kwargs=objfun_kwargs,
            initial_kwargs={'Pm':Pm, 'Em':Em})


