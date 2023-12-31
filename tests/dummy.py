import numpy as np

from hydrodiy.stat.transform import BoxCox2
from hydrodiy.data.containers import Vector
from pygme.model import Model, ParamsVector, UH

from pygme.calibration import Calibration, CalibParamsVector,\
                                        ObjFun, ObjFunSSE

class ObjFunSSEargs(ObjFun):

    def __init__(self):
        super(ObjFunSSEargs, self).__init__('SSE')
        self.BC = BoxCox2()

    def compute(self, obs, sim, **kwargs):
        self.BC.nu = np.nanmean(obs)*1e-3
        idx = kwargs['idx']
        self.BC.lam = kwargs['lam']
        err = self.BC.forward(obs[idx])-self.BC.forward(sim[idx])
        return np.sum(err*err)



class Dummy(Model):

    def __init__(self, checkvalues=None):

        # Config vector
        config = Vector(['continuous'],\
                    [0], [0], [1])

        # params vector
        eps = 1e-7
        vect = Vector(['X1', 'X2'], \
                    [eps, 1], [eps]*2, [10, 10])
        params = ParamsVector(vect, checkvalues=checkvalues)
        params.add_uh('flat', lambda params: params.X1)

        # State vector
        states = Vector(['S1', 'S2'])

        # Model
        Model.__init__(self, 'dummy',
            config, params, states, \
            ninputs=2, \
            noutputsmax=2)

        self.outputs_names = ['a', 'b']

    def initial_fromdata(self, S0=0.):
        self.initialise(states=[S0]*self.states.nval)


    def run(self):
        istart, iend = self.istart, self.iend
        par1, par2 = self.params.values

        kk = range(istart, iend+1)
        outputs = par1+par2*self.inputs[kk, :]

        if np.isclose(self.config['continuous'], 1):
            outputs = outputs + self.states.values[None, :outputs.shape[1]]

        # Write data to selected output indexes
        self.outputs[:istart, :] = np.nan
        self.outputs[kk, :self.noutputs] = outputs[:, :self.noutputs]

        # Store states
        self.states.values = np.concatenate([self.outputs[iend, :],\
                                np.zeros(2-self.noutputs)])



class MassiveDummy(Model):

    def __init__(self):

        config = Vector(['continuous'],\
                    [0], [0], [1])

        params = ParamsVector(Vector(['X1'], \
                    [0], [-10], [10]))

        states = Vector(['S1'])

        Model.__init__(self, 'massivedummy',
            config, params, states, \
            ninputs=2, \
            noutputsmax=2)


    def run(self):
        istart, iend = self.istart, self.iend

        kk = range(istart, iend+1)
        outputs = self.inputs + np.random.uniform(0, 1, (len(kk), 1))
        self.outputs[kk, :] = outputs[kk, :self.noutputs]


class MassiveDummy2(Model):
    ''' Massive Dummy model with no inputs '''

    def __init__(self):

        config = Vector(['continuous'],\
                    [0], [0], [1])

        params = ParamsVector(Vector(['X1'], \
                    [0], [-10], [10]))

        states = Vector(['S1'])

        Model.__init__(self, 'massivedummy2',
            config, params, states, \
            ninputs=0, \
            noutputsmax=2)


    def run(self):
        istart, iend = self.istart, self.iend

        kk = range(istart, iend+1)
        outputs = np.random.uniform(0, 1, (len(kk), 1))
        self.outputs[kk, :] = outputs[kk, :]



class CalibrationDummy(Calibration):

    def __init__(self, warmup, fixed=None, \
                objfun=ObjFunSSE(), \
                objfun_kwargs={}, \
                nplib=2000,\
                checkvalues=None):

        # Calibration params
        model = Dummy(checkvalues=checkvalues)
        cp = Vector(['tX1', 'tX2'], mins=[-10]*2, maxs=[10]*2, \
                defaults=[1, 0])
        calparams = CalibParamsVector(model, cp, \
                            trans2true='exp', \
                            fixed=fixed)

        # Instanciate calibration
        Calibration.__init__(self, calparams, \
            objfun=objfun, \
            warmup=warmup, \
            timeit=True, \
            objfun_kwargs=objfun_kwargs)

        # Parameter library
        params = model.params
        plib = np.random.multivariate_normal(mean=params.defaults, \
                    cov=np.diag((params.maxs-params.mins)/2), \
                    size=nplib)
        plib = np.clip(plib, params.mins, params.maxs)
        self.paramslib = plib

