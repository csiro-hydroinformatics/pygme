import numpy as np

from hydrodiy.data.containers import Vector
from pygme.model import Model

from pygme.calibration import Calibration


class Dummy(Model):

    def __init__(self):

        config = Vector(['continuous'],\
                    [0], [0], [1])

        params = Vector(['X1', 'X2', 'X3'], \
                    [0, 1, 0], [-10, -10, -10], [10, 10, 10])

        states = Vector(['S1', 'S2', 'S3'])

        Model.__init__(self, 'dummy',
            config, params, states, \
            ninputs=2, \
            noutputs=2)

    def run(self, istart, iend, seed=None):

        par1 = self.params[0]
        par2 = self.params[1]
        par3 = self.params[2]

        nval, nvar = self.outputs.shape

        kk = range(istart, iend+1)
        outputs = par1 + par2 *self.inputs[kk, :]

        outputs = np.cumsum(outputs, 0)

        if np.allclose(self.config['continuous'], 1):
            outputs = outputs + self.states

        if par3 > 1e-10:
            outputs *= np.random.uniform(1-par3, 1+par3+1e-20, size=(np.sum(kk), ))

        # Write data to selected output indexes
        self.outputs[kk, :] = outputs[:, :nvar]

        # Store states
        self.states = list(self.outputs[iend, :]) \
                    + [0.] * (2-self.outputs.shape[1])


    def post_params_setter(self):
        uh = np.zeros(self.uh1.nval)
        uh[:4] = 0.25
        self.uh1.values =  uh



class MassiveDummy(Model):

    def __init__(self):

        config = Vector(['continuous'],\
                    [0], [0], [1])

        params = Vector(['X1'], \
                    [0], [-10], [10])

        states = Vector(['S1'])

        Model.__init__(self, 'dummy',
            config, params, states, \
            ninputs=2, \
            noutputs=2)


    def run(self, istart, iend, seed=None):
        kk = range(istart, iend+1)
        outputs = self.inputs + np.random.uniform(0, 1, (len(kk), 1))
        self.outputs[kk, :] = outputs[kk, :]



class CalibrationDummy(Calibration):

    def __init__(self, warmup):
        model = Dummy()

        Calibration.__init__(self,
            nparams=2,
            warmup=warmup,
            model = model,
            timeit = True)

        self._calparams.means =  [1, 0]
        self._calparams.min =  [-10, -10]
        self._calparams.max =  [10, 10]
        self._calparams.covar = [[1, 0.], [0., 20]]


    def cal2true(self, calparams):
        true =  np.array([np.exp(calparams[0]),
            (np.tanh(calparams[1])+1.)*10., 0])

        return true


