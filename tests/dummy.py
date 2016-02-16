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

        self.config.names = 'Config1'

        self.reset()


    def run(self):

        idx_start = self.idx_start
        idx_end = self.idx_end

        par1 = self.params[0]
        par2 = self.params[1]
        par3 = self.params[2]

        nval, nvar = self.outputs.shape

        outputs = par1 + par2 * np.cumsum(self.inputs, 0)

        if par3 > 1e-10:
            outputs[:, 0] *= np.random.uniform(1-par3, 1+par3+1e-20, size=(nval, ))

        self.outputs[idx_start:idx_end+1, :] = outputs[idx_start:idx_end+1, :nvar]

        self.states = list(self.outputs[idx_end, :]) \
                    + [0.] * (2-self.outputs.shape[1])


    def set_uh(self):
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


    def run(self):

        idx_start = self.idx_start
        idx_end = self.idx_end

        nval = self.outputs.shape[0]
        outputs = self.inputs.data + np.random.uniform(0, 1, (nval, 1))
        self.outputs[idx_start:idx_end, :] = outputs[idx_start:idx_end, :]



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


