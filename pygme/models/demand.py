
import numpy as np
import pandas as pd

from pygme.model import Model

class Demand(Model):

    def __init__(self,
            nens_params=1,
            nens_states=1,
            nens_outputs=1):

        Model.__init__(self, 'gr4j',
            nconfig=12,
            ninputs=1,
            nparams=0,
            nstates=2,
            noutputs_max=3,
            nens_params=nens_params,
            nens_states=nens_states,
            nens_outputs=nens_outputs)

        self.config.names = ['demand_{0:2.2d}'.format(m)
                                for m in range(1, 13)]

    def run(self, seed=None):

        _, ninputs, _, _ = self.get_dims('inputs')
        if self._inputs.nvar != ninputs:
            raise ValueError(('Model Demand, self._inputs.nvar({0}) != ' +
                    'self._ninputs({1})').format(
                    self._inputs.nvar, ninputs))

        start, end = self.startend

        ierr = c_pygme_models_demand.demand_run(start, end,
            self._config.data,
            self._inputs.data,
            self._states.data,
            self._outputs.data)

        if ierr > 0:
            raise ValueError(('c_pygme_models_demand.demand_run' +
                ' returns {0}').format(ierr))


