import math
import numpy as np
import pandas as pd

from hydrodiy.data.containers import Vector
from pygme.model import Model, ParamsVector, UH
from pygme.calibration import Calibration, CalibParamsVector, ObjFunBCSSE
from pygme.models.gr4j import gr4j_X1_initial

import c_pygme_models_hydromodels


class GR6J(Model):

    def __init__(self):

        # Config vector
        config = Vector(['nothing'],\
                    [0], [0], [1])

        # params vector
        vect = Vector(['X1', 'X2', 'X3', 'X4', 'X5', 'X6'], \
                    [400, -1, 50, 0.5, 0., 10.], \
                    [1, -50, 1, 0.5, -50., 1], \
                    [1e4, 50, 1e4, 1e2, 50., 1e5])
        params = ParamsVector(vect)

        # UH
        params.add_uh('gr4j_ss1_daily', lambda params: params.X4)
        params.add_uh('gr4j_ss2_daily', lambda params: params.X4)

        # State vector
        states = Vector(['S', 'R', 'A'])

        # Model
        super(GR6J, self).__init__('GR6J',
            config, params, states, \
            ninputs=2, \
            noutputsmax=11)

        self.outputs_names = ['Q', 'ECH', 'AE', \
                    'PR', 'QD', 'QR', 'PERC', 'QExp', 'S', 'R', 'A']


    def initialise_fromdata(self, Pm=0., Em=0.):
        ''' Initialisation of GR6J using
            * Production store: steady state solution from Le Moine
              (2008, Page 212)
            * Routing store: 30% filling level
            * Exponential store: 0% filling level

            Reference:
            Le Moine, Nicolas. "Le bassin versant de surface vu par le souterrain: une voie
            d'amélioration des performances et du réalisme des modèles pluie-débit?." PhD diss., Paris 6, 2008.
        '''
        X1 = self.params.X1
        X3 = self.params.X3

        # Production store
        if Pm > 1e-10 or Em > 1e-10:
            # Using the routine from Le Moine
            S0 = X1 * gr4j_X1_initial(Pm, Em, X1)
        else:
            S0 = 0.5 * X1

        # Routing store
        R0 = 0.3 * X3

        # Model initialisation
        self.initialise(states=[S0, R0, 0.])



    def run(self):
        # Get uh object (not set_timebase function, see ParamsVector class)
        _, uh1 = self.params.uhs[0]
        _, uh2 = self.params.uhs[1]

        # Run gr4j c code
        ierr = c_pygme_models_hydromodels.gr6j_run(uh1.nord, \
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
            raise ValueError(('c_pygme_models_hydromodels.gr6j_run' +
                ' returns {0}').format(ierr))


class CalibrationGR6J(Calibration):

    def __init__(self, objfun=ObjFunBCSSE(0.2), \
                    warmup=5*365, \
                    timeit=False, \
                    fixed=None, \
                    objfun_kwargs={},
                    Pm=0., Em=0.):

        # Input objects for Calibration class
        model = GR6J()
        params = model.params

        # Parameter transformation
        trans2true = lambda x: np.array([
                        math.exp(x[0]), \
                        math.sinh(x[1]), \
                        math.exp(x[2]), \
                        0.49+math.exp(x[3]), \
                        math.sinh(x[4]), \
                        math.exp(x[5])
                    ])

        true2trans = lambda x: np.array([
                        math.log(x[0]), \
                        math.asinh(x[1]), \
                        math.log(x[2]), \
                        math.log(x[3]-0.49), \
                        math.asinh(x[4]), \
                        math.log(x[5])
                    ])

        # initialisation of states
        if Pm < 0 or Em < 0 :
            raise ValueError('Expected Pm and Em >0, '+\
                'got Pm={0}, Em={1}'.format(Pm, Em))

        initial_kwargs = {'Pm':Pm, 'Em':Em}

        # Build calib vector
        cp = Vector(['tX1', 'tX2', 'tX3', 'tX4', 'tX5', 'tX6'], \
                mins=true2trans(params.mins),
                maxs=true2trans(params.maxs),
                defaults=true2trans(params.defaults))

        calparams = CalibParamsVector(model, cp, \
            trans2true=trans2true, \
            true2trans=true2trans,\
            fixed=fixed)

        # Build parameter library from
        # MVT norm in transform space
        tplib = np.random.multivariate_normal(\
                    mean=[5.3, -0.37, 2.5, 0.32, 0.11, 1.5],\
                    cov = \
                        [[2.85701,0.31298,-1.08943,-0.19579,0.11324,-0.60121],\
                        [0.31298,0.40240,-0.51347,-0.13173,0.00879,-0.14192],\
                        [-1.08943,-0.51347,1.99216,-0.03083,-0.03706,0.76333],\
                        [-0.19579,-0.13173,-0.03083,0.55924,0.01843,-0.05714],\
                        [0.11324,0.00879,-0.03706,0.01843,0.63799,0.03747], \
                        [-0.60121,-0.14192,0.76333,-0.05714,0.03747,2.25238]], \
                    size=5000)
        tplib = np.clip(tplib, calparams.mins, calparams.maxs)
        plib = tplib * 0.
        plib[:, [0, 2, 3, 5]] = np.exp(tplib[:, [0, 2, 3, 5]])
        plib[:, 3] += 0.49
        plib[:, [1, 4]] = np.sinh(tplib[:, [1, 4]])

        # Instanciate calibration
        super(CalibrationGR6J, self).__init__(calparams, \
            objfun=objfun, \
            warmup=warmup, \
            timeit=timeit, \
            paramslib=plib, \
            objfun_kwargs=objfun_kwargs, \
            initial_kwargs=initial_kwargs)


