
import numpy as np
import pandas as pd

from calendar import month_abbr as month

import c_pygme_models_basics
from pygme.model import Model


class Clip(Model):

    def __init__(self,
            nens_params=1,
            nens_states=1,
            nens_outputs=1):

        Model.__init__(self, 'clip',
            nconfig=2,
            ninputs=1,
            nparams=0,
            nstates=0,
            noutputs_max=1,
            nens_params=nens_params,
            nens_states=nens_states,
            nens_outputs=nens_outputs)

        self.config.names = ['min', 'max']
        self.config.default = [-np.inf, np.inf]

    def run(self, seed=None):

        start, end = self.startend

        kk = range(start, end+1)
        inputs = self.inputs[kk, 0]

        m1 = self.config['min']
        m2 = self.config['max']
        self.outputs[kk, 0] = np.clip(inputs, m1, m2)



class MonthlyPattern(Model):

    def __init__(self,
            nens_params=1,
            nens_states=1,
            nens_outputs=1):

        Model.__init__(self, 'monthlypattern',
            nconfig=12,
            ninputs=0,
            nparams=0,
            nstates=1,
            noutputs_max=1,
            nens_params=nens_params,
            nens_states=nens_states,
            nens_outputs=nens_outputs)

        self.config.names = [month[m]  for m in range(1, 13)]
        self.config.default = [0.] * 12


    def initialise(self, states=None, statesuh=None):

        # Set default initial states
        if states is None:
            states = [20100101.]

        super(MonthlyPattern, self).initialise(states, statesuh)


    def run(self, seed=None):

        _, ninputs, _, _ = self.get_dims('inputs')
        if self._inputs.nvar != ninputs:
            raise ValueError(('Model {2}, self._inputs.nvar({0}) != ' +
                    'self._ninputs({1})').format(
                    self._inputs.nvar, ninputs, self.name))

        start, end = self.startend

        ierr = c_pygme_models_basics.monthlypattern_run(start, end,
            self.config.data,
            self._states.data,
            self._outputs.data)

        if ierr > 0:
            raise ValueError(('c_pygme_models_basics.monthlypattern_run' +
                ' returns {0}').format(ierr))


