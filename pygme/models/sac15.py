import math
import numpy as np
import pandas as pd

from hydrodiy.data.containers import Vector
from hydrodiy.stat import sutils

from pygme.model import Model, ParamsVector
from pygme.calibration import Calibration, \
                    CalibParamsVector, \
                    ObjFunBiasBCSSE

from pygme import has_c_module
if has_c_module("models_hydromodels"):
    import c_pygme_models_hydromodels

# Transformed parameters mean and covariance
SAC15_TMEAN = np.array([0.25, 3.95, 3.82, -3.05, -2.33, 4.99, -4.99, 0.89, \
                        0.00, -3.17, -0.38, 3.99, -7.40, 3.47, 4.06])

SAC15_TCOV = np.array(\
[	[0.04, -0.0, -0.0, 0.1, 0.0, -0.0, -0.0, -0.0, 0.0, -0.1, 0.0, 0.0, -0.0, 0.0, -0.1],
	[-0.04, 4.8, 2.2, -0.3, 0.9, -0.43, 0.7, -0.56, 0.0, -1.6, -0.1, -0.5, -2.0, -0.1, -2.2],
	[-0.02, 2.2, 3.8, 0.7, -0.1, -0.29, 0.3, -0.47, 0.0, -1.0, 0.0, -0.4, -1.7, 0.1, -2.0],
	[0.07, -0.3, 0.7, 4.6, -1.7, -0.03, -2.1, -0.04, 0.0, 0.0, 0.1, 0.3, -0.7, 0.3, -0.9],
	[0.02, 0.9, -0.1, -1.7, 3.6, -0.22, -0.3, -0.17, 0.0, -0.7, -0.2, -0.1, -1.9, -0.1, -0.9],
	[-0.0, -0.4, -0.3, -0.0, -0.2, 0.58, 0.0, 0.11, 0.0, 0.5, 0.1, 0.3, -0.1, 0.1, 0.3],
	[-0.03, 0.7, 0.3, -2.1, -0.3, 0.02, 9.6, -0.03, 0.0, -1.3, -0.3, -0.3, 1.0, 0.2, 0.6],
	[-0.0, -0.6, -0.5, -0.0, -0.2, 0.11, -0.0, 0.47, 0.0, 0.4, -0.0, 0.1, 1.2, -0.0, 0.5],
	[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
	[-0.08, -1.6, -1.0, 0.0, -0.7, 0.52, -1.3, 0.42, 0.0, 7.6, 0.5, 0.9, 0.1, 0.3, 1.5],
	[0.03, -0.1, 0.0, 0.1, -0.2, 0.09, -0.3, -0.0, 0.0, 0.5, 2.7, 0.1, -0.5, 0.1, -0.0],
	[0.01, -0.5, -0.4, 0.3, -0.1, 0.28, -0.3, 0.11, 0.0, 0.9, 0.1, 2.6, -0.4, 0.1, 0.2],
	[-0.05, -2.0, -1.7, -0.7, -1.9, -0.11, 1.0, 1.24, 0.0, 0.1, -0.5, -0.4, 89.9, -0.3, 6.3],
	[0.02, -0.1, 0.1, 0.3, -0.1, 0.06, 0.2, -0.03, 0.0, 0.3, 0.1, 0.1, -0.3, 1.1, -0.2],
	[-0.06, -2.2, -2.0, -0.9, -0.9, 0.35, 0.6, 0.54, 0.0, 1.5, -0.0, 0.2, 6.3, -0.2, 6.7],
])

def logit_fwd(x):
    u = max(1e-8, min(1-1e-8, x))
    return math.log(u/(1-u))

def logit_inv(tx):
    return 1/(1+math.exp(-tx))

# Transformation functions for sac parameters
def sac15_trans2true(tx):
    x = np.exp(tx)
    x[0] = tx[0]                   # ADIMP
    x[3] = logit_inv(tx[3])        # LZPK
    x[4] = logit_inv(tx[4])        # LZSK
    x[6] = (9.99+tx[6])/19.98      # PFREE
    x[8] = tx[8]                   # SARVA
    x[9] = math.sinh((5+tx[9])/10.)# SIDE
    x[12] = logit_inv(tx[12])        # UZK
    return x

def sac15_true2trans(x):
    tx = np.log(np.maximum(1e-10, x))
    tx[0] = x[0]                    # ADIMP
    tx[3] = logit_fwd(x[3])         # LZPK
    tx[4] = logit_fwd(x[4])         # LZSK
    tx[6] = x[6]*19.98-9.99         # PFREE
    tx[8] = x[8]                    # SARVA
    tx[9] = 10.*math.asinh(x[9])-5. # SIDE
    tx[12] = logit_fwd(x[12])       # UZK

    return tx


class SAC15(Model):

    def __init__(self):
        config = Vector(["nodata"], [0], [0], [1])

        # Param vector
        #defaults = [0.1, 82., 32., 0.04, 0.24, 179.,
        #                    0.4, 1.81, 0.01, 0., 0., 49., 0.4, 76., 60.], \
        #mins = [1e-5, 1e-2, 1e-2, 1e-3, 1e-3, 10.,
        #                    1e-2, 1., 0., -0.5, 0, 1e-1, 1e-5, 1., 1e-2], \
        #maxs = [0.9, 1e3, 1e3, 0.9, 0.9, 1e3, 0.5,
        #                    10., 0.2, 0.5, 10., 2e3, 1-1e-10, 6e2, 2e3]

        defaults = [
        		0.25, #Adimp
        		249.67, #Lzfpm
        		199.33, #Lzfsm
        		0.14, #Lzpk
        		0.19, #Lzsk
        		203.39, #Lztwm
        		0.25, #Pfree
        		3.06, #Rexp
        		0.00, #Sarva
        		0.19, #Side
        		1.17, #Lag
        		207.43, #Uzfwm
        		0.18, #Uzk
        		51.19, #Uztwm
        		459.09, #Zperc
        ]

        mins = [
		        0.00, #Adimp
		        0.01, #Lzfpm
		        0.23, #Lzfsm
		        0.00, #Lzpk
		        0.00, #Lzsk
		        10.00, #Lztwm
		        0.01, #Pfree
		        1.00, #Rexp
		        0.00, #Sarva
		        -0.50, #Side
		        0.00, #Lag
		        4.00, #Uzfwm
		        0.00, #Uzk
		        1.00, #Uztwm
		        0.10, #Zperc
        ]

        maxs = [
		        0.90, #Adimp
		        1500.00, #Lzfpm
		        1500.00, #Lzfsm
		        0.90, #Lzpk
		        0.90, #Lzsk
		        1500.00, #Lztwm
		        0.50, #Pfree
		        7.00, #Rexp
		        0.00, #Sarva
		        0.50, #Side
		        10.00, #Lag
		        2000.00, #Uzfwm
		        1.00, #Uzk
		        400.00, #Uztwm
		        2000.00, #Zperc
        ]

        vect = Vector(['Adimp', 'Lzfpm', 'Lzfsm', 'Lzpk', \
                    'Lzsk', 'Lztwm', 'Pfree', 'Rexp', 'Sarva', \
                    'Side', 'Lag', 'Uzfwm', 'Uzk', 'Uztwm', 'Zperc'], \
                    defaults=defaults,
                    mins=mins, \
                    maxs=maxs)
        params = ParamsVector(vect)

        # UH
        params.add_uh("lag", lambda params: params.Lag)

        # State vector
        states = Vector(["Uztwc", "Uzfwc", "Lztwc", \
                    "Lzfsc", "Lzfpc", "Adimc"], check_bounds=False)

        # Model
        super(SAC15, self).__init__("SAC15", \
                config, params, states, \
                ninputs=2, \
                noutputsmax=11)

    def initialise_fromdata(self):
        # Initialise stores
        Lzfpm = self.params.Lzfpm
        Lzfsm = self.params.Lzfsm
        Lztwm = self.params.Lztwm
        Uzfwm = self.params.Uzfwm
        Uztwm = self.params.Uztwm

        self.initialise(states= [Uztwm, Uzfwm/2, Lztwm/2,
                Lzfsm/2, Lzfpm/2, Uztwm+Lztwm/2])

    def run(self):
        # Get uh object
        _, uh = self.params.uhs[0]

        ierr = c_pygme_models_hydromodels.sac15_run(uh.nord,\
            self.istart, self.iend,\
            self.params.values,\
            uh.ord,\
            self.inputs, \
            uh.states, \
            self.states.values,
            self.outputs)

        if ierr > 0:
            raise ValueError(('c_pygme_models_hydromodels.sac15_run' +
                ' returns {0}').format(ierr))



class CalibrationSAC15(Calibration):
    # Fix Sarva to 0 because it seems useless
    def __init__(self, objfun=ObjFunBiasBCSSE(0.5), \
                    warmup=5*365, \
                    timeit=False, \
                    fixed={"Sarva": 0.}, \
                    objfun_kwargs={}, \
                    nparamslib=20000):

        # Input objects for Calibration class
        model = SAC15()
        params = model.params

        # Calib param vector
        cp = Vector(['tAdimp', 'tLzfpm', 'tLzfsm', 'tLzpk', \
                    'tLzsk', 'tLztwm', 'tPfree', 'tRexp', 'tSarva', \
                    'tSide', 'tLag', 'tUzfwm', 'tUzk', 'tUztwm', 'tZperc'], \
                        mins=sac15_true2trans(params.mins),
                        maxs=sac15_true2trans(params.maxs),
                        defaults=sac15_true2trans(params.defaults))

        calparams = CalibParamsVector(model, cp, \
            trans2true=sac15_trans2true, \
            true2trans=sac15_true2trans,\
            fixed=fixed)

        # Instanciate calibration
        super(CalibrationSAC15, self).__init__(calparams, \
            objfun=objfun, \
            warmup=warmup, \
            timeit=timeit, \
            objfun_kwargs=objfun_kwargs)

        # Build parameter library from
        # MVT norm in transform space using latin hypercube
        tplib = sutils.lhs_norm(nparamslib, SAC15_TMEAN, SAC15_TCOV)

        # Back transform
        plib = tplib * 0.
        for i in range(len(plib)):
            plib[i, :] = sac15_trans2true(tplib[i, :])
        plib = np.clip(plib, model.params.mins, model.params.maxs)
        self.paramslib = plib




