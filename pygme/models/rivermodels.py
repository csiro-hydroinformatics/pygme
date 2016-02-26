
import numpy as np
import pandas as pd

from calendar import month_abbr as month

import c_pygme_models_demand
from pygme.model import Model

class Demand(Model):

    def __init__(self,
            nens_params=1,
            nens_states=1,
            nens_outputs=1):

        Model.__init__(self, 'demand',
            nconfig=12,
            ninputs=2,
            nparams=0,
            nstates=2,
            noutputs_max=3,
            nens_params=nens_params,
            nens_states=nens_states,
            nens_outputs=nens_outputs)

        self.config.names = [month[m]  for m in range(1, 13)]
        self.config.default = [0.] * 12


    def initialise(self, states=None, statesuh=None):

        # Set default initial states
        if states is None:
            states = [20100101., 0.]

        super(Demand, self).initialise(states, statesuh)



    def run(self, seed=None):

        _, ninputs, _, _ = self.get_dims('inputs')
        if self._inputs.nvar != ninputs:
            raise ValueError(('Model Demand, self._inputs.nvar({0}) != ' +
                    'self._ninputs({1})').format(
                    self._inputs.nvar, ninputs))

        start, end = self.startend

        ierr = c_pygme_models_demand.demand_run(start, end,
            self.config.data,
            self._inputs.data,
            self._states.data,
            self._outputs.data)

        if ierr > 0:
            raise ValueError(('c_pygme_models_demand.demand_run' +
                ' returns {0}').format(ierr))


