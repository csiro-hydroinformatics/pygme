
import numpy as np
import pandas as pd

from calendar import month_abbr as month

import c_pygme_models_patterns
from pygme.model import Model


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


    def runblock(self, istart, iend, seed=None):

        ierr = c_pygme_models_patterns.monthlypattern_run(istart, iend,
            self.config.data,
            self._states.data,
            self._outputs.data)

        if ierr > 0:
            raise ValueError(('c_pygme_models_patterns.monthlypattern_run' +
                ' returns {0}').format(ierr))


class SinusPattern(Model):

    def __init__(self,
            nens_params=1,
            nens_states=1,
            nens_outputs=1):

        Model.__init__(self, 'sinuspattern',
            nconfig=5,
            ninputs=0,
            nparams=4,
            nstates=2,
            noutputs_max=1,
            nens_params=nens_params,
            nens_states=nens_states,
            nens_outputs=nens_outputs)

        self.config.names = ['is_cumulative', 'year_monthstart',
                'vmin', 'vmax', 'startdate']
        self.config.default = [0., 1., 0., 1e3, 20010101.]
        self.config.reset()

        self._params.names = ['lower', 'upper', 'phi', 'nu']
        self._params.default = [0., 1., 0., 0.]
        self._params.min = [0., 0., 0., -6.]
        self._params.max = [1., 1., 1., 6.]

        self.reset()


    def initialise(self, states=None, statesuh=None):

        # Set default initial states
        if states is None:
            states = [self.config['startdate'], 0.]

        super(SinusPattern, self).initialise(states, statesuh)


    def runblock(self, istart, iend, seed=None):

        ierr = c_pygme_models_patterns.sinuspattern_run(istart, iend,
            self.config.data,
            self._params.data,
            self._states.data,
            self._outputs.data)

        if ierr > 0:
            raise ValueError(('c_pygme_models_patterns.sinuspattern_run' +
                ' returns {0}').format(ierr))

