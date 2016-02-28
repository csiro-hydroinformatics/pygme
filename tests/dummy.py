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
            run_as_block=True,
            nens_params=nens_params,
            nens_states=nens_states,
            nens_outputs=nens_outputs)

        self._params.default = [0., 1., 0.]
        self._params.min = [-10., -10., -10.]
        self._params.max = [10., 10., 10.]

        self.config.names = 'continuous'
        self.config.default = 1
        self.config.data = 1


    def runblock(self, istart, iend, seed=None):

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
            run_as_block=True,
            nens_params=nens_params,
            nens_states=nens_states,
            nens_outputs=nens_outputs)


    def runblock(self, istart, iend, seed=None):

        kk = range(istart, iend+1)
        outputs = self.inputs + np.random.uniform(0, 1, (len(kk), 1))
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


