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
        self.config.reset(0.)

        self.reset()


    def run(self, seed=None):

        idx_start = self.idx_start
        idx_end = self.idx_end

        nval, nvar = self.outputs.shape
        par1, par2, par3 = self.params

        kk = np.arange(idx_start, idx_end+1)
        outputs = par2 * np.cumsum(par1 + self.inputs[kk, :], 0)

        if np.allclose(self.config['continuous'], 1):
            outputs = self.states + outputs

        if par3 > 1e-10:
            outputs[:, 0] *= np.random.uniform(1-par3, 1+par3+1e-20,
                                size=(len(kk), ))

        self.outputs[idx_start:idx_end+1, :] = outputs[:, :nvar]

        self.states = list(self.outputs[idx_end, :]) \
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

        idx_start = self.idx_start
        idx_end = self.idx_end
        kk = np.arange(idx_start, idx_end+1)

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


