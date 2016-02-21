import numpy as np

from pygme.model import Model

from pygme.calibration import Calibration


class Dummy(Model):

    def __init__(self,
            nens_params=1,
            nens_states=1,
            nens_outputs=1):

        Model.__init__(self, 'dummy',
            nconfig=1,\
            ninputs=2, \
            nparams=3, \
            nstates=2, \
            noutputs_max=2,
            nens_params=nens_params,
            nens_states=nens_states,
            nens_outputs=nens_outputs)

        self._params.default = [0., 1., 0.]
        self._params.min = [-10., -10., -10.]
        self._params.max = [10., 10., 10.]

        self.config.names = 'continuous'
        self.config.default = 1


    def run(self, seed=None):

        index = self._inputs.index
        index_start = self.index_start
        index_end = self.index_end

        par1 = self.params[0]
        par2 = self.params[1]
        par3 = self.params[2]

        nval, nvar = self.outputs.shape

        outputs = par1 + par2 *self.inputs

        if np.allclose(self.config['continuous'], 1):
            outputs = outputs + self.states

        outputs = np.cumsum(outputs, 0)

        if par3 > 1e-10:
            outputs[:, 0] *= np.random.uniform(1-par3, 1+par3+1e-20, size=(nval, ))

        # Write data to selected output indexes
        kk = (index >= index_start) & (index <= index_end)
        self.outputs[kk, :] = outputs[kk, :nvar]

        # Store states
        kk = np.where(index == index_end)[0]
        self.states = list(self.outputs[kk, :]) \
                    + [0.] * (2-self.outputs.shape[1])

    def post_params_setter(self):
        uh = np.zeros(self._uh.nval)
        uh[:4] = 0.25
        self.uh =  uh



class MassiveDummy(Model):

    def __init__(self,
            nens_params=1,
            nens_states=1,
            nens_outputs=1):

        Model.__init__(self, 'dummy',
            nconfig=0,
            ninputs=1,
            nparams=0,
            nstates=0,
            noutputs_max=1,
            nens_params=nens_params,
            nens_states=nens_states,
            nens_outputs=nens_outputs)


    def run(self, seed=None):

        nval = self.outputs.shape[0]
        outputs = self.inputs + np.random.uniform(0, 1, (nval, 1))

        index_start = self.index_start
        index_end = self.index_end
        kk = np.arange(index_start, index_end+1)

        self.outputs[kk, :] = outputs[kk, :]



class CalibrationDummy(Calibration):

    def __init__(self):
        model = Dummy()

        Calibration.__init__(self,
            model = model, \
            ncalparams = 2, \
            timeit = True)

        self._calparams.means =  [1, 0]
        self._calparams.min =  [-10, -10]
        self._calparams.max =  [10, 10]
        self._calparams.covar = [[1, 0.], [0., 20]]


    def cal2true(self, calparams):
        true =  np.array([np.exp(calparams[0]), (np.tanh(calparams[1])+1.)*10., 0])
        return true


