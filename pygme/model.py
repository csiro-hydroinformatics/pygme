import math, itertools
import copy
import random
import numpy as np

import c_pygme_models_utils

NUHMAXLENGTH = c_pygme_models_utils.uh_getnuhmaxlength()

from hydrodiy.data.containers import Vector

# Overload Vector class to handle post-setter of UH
class ParamsVector(Vector):

    def __init__(self, params, post_setter_args):
        Vector.__init__(self, params.names, \
            params.defaults, params.mins, params.maxs, \
            params.hitbounds, \
            post_setter_args=post_setter_args)

    def post_setter(self, model):
        model.post_params_setter()


class Model(object):

    def __init__(self, name, config, params, states, \
            ninputs, noutputs, nuh=2):

        # Model name
        self.name = name

        # Config and params vectors
        self.config = config
        self.params = ParamsVector(params, (self, ))
        self.states = states

        # Dimensions
        self._ninputs = int(ninputs)
        self._noutputs = int(ninputs)

        # data
        self._inputs = None
        self._outputs = None

        # UH ordinates and states
        self.nuh = nuh
        uh_names = ['UH'+str(k) for k in range(1, NUHMAXLENGTH+1)]

        for iuh in range(1, nuh+1):
            setattr(self, 'uh'+str(iuh),
                Vector(uh_names, \
                    defaults=np.zeros(NUHMAXLENGTH), \
                    mins=np.zeros(NUHMAXLENGTH), \
                    maxs=np.ones(NUHMAXLENGTH)))

            # All values in [0, +inf]
            setattr(self, 'statesuh'+str(iuh),
                Vector(uh_names, \
                    defaults=np.zeros(NUHMAXLENGTH), \
                    mins=np.zeros(NUHMAXLENGTH)))


    def __str__(self):
        str = ('\n{0} model implementation\n'+\
            '\tConfig: {1}\n\tParams: {2}\b\tStates: {3}\n\tNUH: {4}').format( \
            self.name, self.config.names, self.params.names, \
            self.states.names, self.nuh)
        return str


    def __uh_setter(self, iuh, values):
        if np.abs(np.sum(values)-1.) > 1e-9:
            raise ValueError('Model {0}: Expected sum uh{1} = 1, got {2}'.format(\
                                self.name, iuh, np.sum(values)))
        uh = getattr(self, 'uh'+str(iuh))
        uh.values = values


    @property
    def ninputs(self):
        return self._ninputs


    @property
    def noutputs(self):
        return self._noutputs


    @property
    def inputs(self):
        return self._inputs


    @property
    def outputs(self):
        return self._outputs


    def post_params_setter(self):
        pass


    def allocate(self, inputs, noutputs=1):
        ''' We define the number of outputs here to allow more flexible memory allocation '''

        if noutputs <= 0 or noutputs > self.noutputs:
            raise ValueError(('model {0}: ' +\
                'Expected noutputs in [1, {1}], got {2}'.format(\
                    self.name, self.noutputs, noutputs)))

        # Allocate inputs
        inputs = np.atleast_2d(inputs)
        if inputs.shape[1] != self._ninputs:
            raise ValueError(('model {0}: Expected ninputs={1}, '+\
                                'got {2}').format(self.name, self._ninputs, \
                                inputs.shape[1]))
        self._inputs = inputs

        # Allocate outputs
        self._outputs = np.zeros((inputs.shape[0], noutputs))


    def initialise(self, states=None, *args):
        ''' Initialise state vector and potentially all UH states vectors '''
        if states is None:
            self.states.reset()
        else:
            self.states.values = states

        if len(args)>0:
            for iuh in range(1, self.nuh+1):
                suh = getattr(self, 'statesuh{0}'.format(iuh))
                if iuh<=len(args):
                    suh.reset()
                else:
                    suh.values = args[iuh]


    def run(self, istart, iend, seed):
        ''' Run the model '''
        raise NotImplementedError(('model {0}: '+\
            'Method run not implemented').format(self.model))


    def clone(self):
        ''' Clone the current object model '''
        return copy.deepcopy(self)


