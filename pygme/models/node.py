
import numpy as np
import pandas as pd


from pygme.model import Model


class NodeModel(Model):

    def __init__(self,
            ninputs=1, noutputs=1,
            nens_params=1,
            nens_states=1,
            nens_outputs=1):

        Model.__init__(self, 'nodemodel',
            nconfig=3,
            ninputs=ninputs,
            nparams=noutputs,
            nstates=0,
            noutputs_max=noutputs,
            nens_params=nens_params,
            nens_states=nens_states,
            nens_outputs=nens_outputs)

        self.config.names = ['min', 'max', 'is_conservative']
        self.config.default = [-np.inf, np.inf, 1.]
        self.config.reset()

        self._params.names = ['F{0}'.format(i) for i in range(noutputs)]
        self._params.default = [1.] * noutputs
        self._params.reset()


    def runblock(self, istart, iend, seed=None):

        kk = range(istart, iend+1)
        nval = len(kk)

        # Sum of inputs and clip
        m1 = self.config['min']
        m2 = self.config['max']
        inputs = np.clip(self.inputs[kk, :].sum(axis=1), m1, m2)

        _, noutputs, _, _ = self.get_dims('outputs')

        if noutputs > 1:
            # Compute split parameters
            params = self.params
            if self.config['is_conservative'] == 1:
                params = params/np.sum(params)
            params = np.diag(params)

            # Split to ouputs
            outputs = np.dot(np.repeat(inputs.reshape((nval, 1)),
                noutputs, axis=1), params)
            self.outputs[kk, :] = outputs
        else:
            self.outputs[kk, 0] = inputs


