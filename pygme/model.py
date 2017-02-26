import math, itertools
import copy
import random
import numpy as np

import c_pygme_models_utils

NUHMAXLENGTH = c_pygme_models_utils.uh_getnuhmaxlength()

from hydrodiy.data.containers import Vector


class Model(object):

    def __init__(self, name, \
            config_names, \
            params_names, \
            states_names, \
            ninputs, \
            noutputs):

        # Model name
        self.name = name

        # Vectors
        self._config = Vector(config_names)
        self._params = Vector(params_names)
        self._states = Vector(states_names)

        # Dimensions
        self._ninputs = int(ninputs)
        self._noutputs = int(ninputs)

        # data
        self._inputs = None
        self._outputs = None

        # UH ordinates and states
        uh_names = ['UH'+str(k) for k in range(1, NUHMAXLENGTH+1)]
        for iuh in range(1, 3):
            # All values in [0, 1]
            setattr(self, '_uh'+str(iuh),
                Vector(uh_names, \
                    defaults=np.zeros(NUHMAXLENGTH), \
                    mins=np.zeros(NUHMAXLENGTH), \
                    maxs=np.ones(NUHMAXLENGTH))

            # All values in [0, +inf]
            setattr(self, '_statesuh'+str(iuh),
                Vector(uh_names, \
                    defaults=np.zeros(NUHMAXLENGTH), \
                    mins=np.zeros(NUHMAXLENGTH))


    def __str__(self):
        str = '\n{0} model implementation\n'.format( \
            self.name)
        return str


    @property
    def params_names(self):
        return self._params.names

    @property
    def params(self):
        return self._params.values

    @params.setter
    def params(self, value):
        self._params.values = value

        # When setting params, applies post-processing
        # (e.g. UH setting)
        self.post_params_setter()


    def __uh_setter(self, iuh, value):
        if np.abs(np.sum(value)-1.) > 1e-9:
            raise ValueError('Model {0}: Expected sum uh{1} = 1, got {2}'.format(\
                                self.name, iuh, np.sum(value)))
        uh = getattr(self, '_uh'+str(iuh))
        uh.values = value

    @property
    def uh1(self):
        return self._uh1.values

    @uh1.setter
    def uh1(self, value):
        self.__uh_setter(1, value)


    @property
    def uh2(self):
        return self._uh2.values

    @uh2.setter
    def uh2(self, value):
        self.__uh_setter(2, value)


    @property
    def statesuh1(self):
        return self._statesuh1.values

    @statesuh1.setter
    def statesuh1(self, value):
        self._statesuh1.values = value


    @property
    def statesuh2(self):
        return self._statesuh2.values

    @statesuh2.setter
    def statesuh2(self, value):
        self._statesuh2.values = value


    @property
    def states_names(self):
        return self._states.names

    @property
    def states(self):
        return self._states.values

    @states.setter
    def states(self, value):
        self._states.values = value


    @property
    def config_names(self):
        return self._states.names

    @property
    def config(self):
        return self._states.values

    @config.setter
    def config(self, value):
        self._config.values = value


    @property
    def inputs(self):
        return self._inputs


    @property
    def outputs(self):
        return self._outputs


    def allocate(self, inputs, noutputs=1):
        ''' We define the number of outputs here to allow more flexible memory allocation '''

        if noutputs <= 0 or noutputs > self.noutputs:
            raise ValueError(('model {0}: ' +\
                'Expected noutputs in [1, {1}], got {2}'.format(\
                    self.name, self.noutputs, noutputs))

        # Allocate inputs
        inputs = np.atleast_2d(inputs)
        if inputs.shape[1] != self._ninputs:
            raise ValueError(('model {0}: Expected ninputs={1}, '+\
                                'got {2}').format(self.name, self._ninputs, \
                                inputs.shape[1]))
        self._inputs = inputs

        # Allocate outputs
        self._outputs = np.zeros((inputs.shape[0], noutputs))


    def post_params_setter(self):
        pass


    def initialise(self, states=None, statesuh1=None, \
                    statesuh2=None):

        if self._states is None:
            raise ValueError(('With model {0}, Cannot initialise when'+
                ' states is None. Please allocate').format(self.name))

        if states is None:
            self._states.reset()
        else:
            self._states.values = states

        if statesuh1 is None:
            self._statesuh1.reset()
        else:
            self._statesuh1.values = statesuh1

        if statesuh2 is None:
            self._statesuh2.reset()
        else:
            self._statesuh2.values = statesuh1


    def run(self, runblock=False):
        ''' Run the model '''

        if self._inputs is None or self._outputs is None:
            raise ValueError('model {0}: '+\
                'inputs or outputs are None'.format(self.name))

        if runblock:
            # Run model in block mode
            self.runblock(istart, iend)

        else:
            # Run model in time step mode over the range
            # [istart - iend]
            for i in np.arange(istart, iend+1):
                self.runtimestep(i)


    def runblock(self, istart, iend):
        raise NotImplementedError(('model {0}: '+\
            'Method runblock not implemented').format(self.model))

    def runtimestep(self, istep):
        self.runblock(istep, istep)


    def clone(self):
        ''' Clone the current object model '''
        return copy.deepcopy(self)


