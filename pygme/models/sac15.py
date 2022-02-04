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
SAC15_TMEAN = np.array([0.25, 3.64, 3.51, -3.32, -2.85, 5.08, -3.88, 0.83, 0.00, -2.57, \
                        -0.59, 3.98, -5.72, 3.37, 4.37])
SAC15_TCOV = np.array([
	[0.04, -0.07, -0.05, 0.08, 0.01, 0.02, 0.01, -0.0, -0.00014, -0.01, 0.09, 0.03, -0.22, 0.03, -0.06],
	[-0.07, 3.89, 1.43, -0.72, 1.57, -0.53, 0.05, -0.47, 0.00123, -1.45, -0.82, -0.4, -0.39, -0.32, -1.92],
	[-0.05, 1.43, 2.52, 0.98, -0.35, -0.26, -0.41, -0.23, 0.0005, -0.69, -0.48, -0.4, -0.23, -0.15, -1.49],
	[0.08, -0.72, 0.98, 4.86, -2.23, 0.02, -1.94, -0.06, -0.00078, 0.45, 0.44, 0.08, -0.79, 0.21, -1.37],
	[0.01, 1.57, -0.35, -2.23, 3.84, -0.26, 0.11, -0.23, 0.00066, -1.02, -0.91, 0.02, -1.1, -0.2, -0.9],
	[0.02, -0.53, -0.26, 0.02, -0.26, 0.47, -0.08, 0.08, -0.00025, 0.51, 0.21, 0.22, -0.25, 0.14, 0.34],
	[0.01, 0.05, -0.41, -1.94, 0.11, -0.08, 8.06, 0.16, 0.00213, -0.52, -0.72, 0.02, -1.0, 0.28, 0.86],
	[-0.0, -0.47, -0.23, -0.06, -0.23, 0.08, 0.16, 0.41, -0.00028, 0.35, 0.09, 0.02, 1.07, 0.0, 0.47],
	[-0.0, 0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 1e-05, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0],
	[-0.01, -1.45, -0.69, 0.45, -1.02, 0.51, -0.52, 0.35, -0.00086, 6.07, 0.84, 0.65, -0.65, 0.51, 1.03],
	[0.09, -0.82, -0.48, 0.44, -0.91, 0.21, -0.72, 0.09, -0.0023, 0.84, 6.56, 0.26, -1.28, 0.33, -0.17],
	[0.03, -0.4, -0.4, 0.08, 0.02, 0.22, 0.02, 0.02, -0.0005, 0.65, 0.26, 2.07, -1.19, 0.16, 0.18],
	[-0.22, -0.39, -0.23, -0.79, -1.1, -0.25, -1.0, 1.07, -1e-05, -0.65, -1.28, -1.19, 76.91, -1.21, 4.59],
	[0.03, -0.32, -0.15, 0.21, -0.2, 0.14, 0.28, 0.0, -0.0003, 0.51, 0.33, 0.16, -1.21, 1.03, -0.03],
	[-0.06, -1.92, -1.49, -1.37, -0.9, 0.34, 0.86, 0.47, 1e-05, 1.03, -0.17, 0.18, 4.59, -0.03, 4.97],
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
        		141.87, #Lzfpm
        		99.09, #Lzfsm
        		0.13, #Lzpk
        		0.14, #Lzsk
        		202.70, #Lztwm
        		0.31, #Pfree
        		2.80, #Rexp
        		0.00, #Sarva
        		0.25, #Side
        		1.18, #Lag
        		176.11, #Uzfwm
        		0.18, #Uzk
        		45.60, #Uztwm
        		446.74 #Zperc
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
        		0.10 #Zperc
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
        		0.05, #Sarva
        		0.50, #Side
        		10.00, #Lag
        		2000.00, #Uzfwm
        		1.00, #Uzk
        		400.00, #Uztwm
        		2000.00 #Zperc
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




