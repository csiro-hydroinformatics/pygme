import math, itertools
import copy
import random
import numpy as np

import c_pygme_models_utils

NUHMAXLENGTH = c_pygme_models_utils.uh_getnuhmaxlength()

from hydrodiy.data.containers import Vector


UHNAMES = ['gr4j_ss1_daily', 'gr4j_ss2_daily', \
            'gr4j_ss1_hourly', 'gr4j_ss2_hourly', \
            'lag', 'triangle', 'flat']

class UH(object):

    def __init__(self, name, iparam=0, nuhmax=NUHMAXLENGTH):

        # Set max length of uh
        if nuhmax > NUHMAXLENGTH or nuhmax<=0:
            raise ValueError(('Expected nuhmax in [1, {0}], '+\
                    'got {1}').format(NUHMAXLENGTH, nuhmax))

        self._nuhmax = nuhmax

        # Check name
        self._uhid = 0
        for uid in range(len(UHNAMES)):
            if name == UHNAMES[uid]:
                self._uhid = uid+1
                break

        if self._uhid == 0:
            expected = '/'.join(UHNAMES)
            raise ValueError('Expected UH name in {0}, got {1}'.format(\
                expected, name))
        self.name = name

        # Initialise ordinates and states
        self._ord = np.zeros(nuhmax, dtype=np.float64)
        self._ord[0] = 1.
        self._states = np.zeros(nuhmax, dtype=np.float64)

        # Index in the model parameter vector
        self._imodelparam = np.int32(iparam)

        # set param value
        self._param = 0.

        # Number of ordinates
        # nuh is stored as an array to be passed to the C routine
        self._nuh = np.array([1], dtype=np.int32)


    def __str__(self):
        str = 'UH {0}: param={1} nuh={2}'.format(self.name, \
                self.param, self.nuh)
        return str


    @property
    def uhid(self):
        return self._uhid


    @property
    def nuhmax(self):
        return self._nuhmax


    @property
    def nuh(self):
        # There is a trick here, we return an
        # integer but the internal state is an array
        return self._nuh[0]


    @property
    def imodelparam(self):
        return self._imodelparam


    @property
    def param(self):
        return self._param

    @param.setter
    def param(self, value):
        # Check value
        value = np.float64(value)

        # Populate the uh ordinates
        ierr = c_pygme_models_utils.uh_getuh(self.nuhmax, self.uhid, \
                                        value, self._nuh, self._ord)
        if ierr>0:
            raise ValueError(('When setting param to {0} for UH {1}, '+\
                'c_pygme_models_utils.uh_getuh returns {2}').format(\
                        value, self.name, ierr))

        # Store parameter value
        self._param = value

        # Reset uh states to a vector of zeros
        # with length _nuh[0]
        self._states[:self._nuh[0]] = 0
        self._ord[self._nuh[0]:] = 0


    @property
    def ord(self):
        return self._ord


    @property
    def states(self):
        return self._states

    @states.setter
    def states(self, values):
        values = np.atleast_1d(values).astype(np.float64)

        nuh = self.nuh
        if values.shape[0]<nuh:
            raise ValueError('Expected state vector to be of length'+\
                '>={0}, got {1}'.format(nuh, values.shape[0]))

        self.reset()
        self._states[:nuh] = values[:nuh]


    def reset(self):
        self._states = np.zeros(self.nuhmax)



# Overload Vector class to change UH and corresponding states
# when changing model parameters
class ParamsVector(Vector):

    def __init__(self, params, uhs=None):

        # Initialise Vector object
        Vector.__init__(self, params.names, \
            params.defaults, params.mins, params.maxs, \
            params.hitbounds)

        if not uhs is None:
            # Check param number set in uhs
            for iuh, uh in enumerate(uhs):
                if uh.imodelparam>=self.nval or uh.imodelparam<0:
                    raise ValueError(('Expected uhs[{0}].iparam in [0, {1}[,'+\
                                    ' got {2}').format(iuh, self.nval, \
                                    uh.imodelparam))

            # Store unit hydrograhs objects
            # (to be modified when parameter values change)
            self._nuh = len(uhs)
            self._uhs = uhs
        else:
            self._uhs = None
            self._nuh = 0


    def __setitem__(self, key, value):
        # Set item for the vector
        Vector.__setitem__(self, key, value)

        # Set UH parameter
        ip = self.__findname__(key)
        if self.nuh>0:
            for uh in self.uhs:
                if uh.imodelparam == ip:
                    uh.param = value
                    break

    @property
    def nuh(self):
        return self._nuh


    @property
    def uhs(self):
        return self._uhs


    @property
    def iuhparams(self):
        return self._iuhparams


    @Vector.values.setter
    def values(self, val):
        # Run the vector value setter
        Vector.values.fset(self, val)

        # Set UH parameter
        if self.nuh>0:
            for uh in self.uhs:
                uh.param = self.values[uh.imodelparam]



class Model(object):

    def __init__(self, name, config, params, states, \
            ninputs, noutputsmax):

        # Model name
        self.name = name

        # Config and params vectors
        self._config = config
        self._params = params
        self._states = states

        # Dimensions
        self._ninputs = int(ninputs)
        self._noutputsmax = int(noutputsmax)
        self._noutputs = 0 # will be set to >0 when outputs are allocated

        # data
        self._inputs = None
        self._outputs = None

        # Start/end index
        self._istart = None
        self._iend = None


    def __str__(self):
        str = ('\n{0} model implementation\n'+\
                '\tConfig: {1}\n\tParams: {2}\b\tStates: {3}'+\
                '\n\tNUH: {4}').format( \
                    self.name, self.config.names, self.params.names, \
                    self.states.names, self.params.nuh)
        return str


    @property
    def params(self):
        return self._params


    @property
    def config(self):
        return self._config


    @property
    def states(self):
        return self._states


    @property
    def ninputs(self):
        return self._ninputs


    @property
    def ntimesteps(self):
        if self._inputs is None:
            raise ValueError('Trying to get ntimesteps, but inputs '+\
                        'are not allocated. Please allocate')

        return self.inputs.shape[0]


    @property
    def istart(self):
        if self._inputs is None:
            raise ValueError('Trying to get istart, '+\
                'but inputs are not allocated. Please allocate')

        if self._istart is None:
            raise ValueError('Trying to get istart, '+\
                'but it is not set. Please set value')

        return self._istart


    @istart.setter
    def istart(self, value):
        ''' Set data '''
        value = np.int32(value)

        if self._inputs is None:
            raise ValueError('Trying to set istart, '+\
                'but inputs are not allocated. Please allocate')

        if value<0 or value>self.ntimesteps-1:
            raise ValueError('Expected istart in [0, {0}], got {2}'.format(\
                self.ntimestep-1, value))

        self._istart = value


    @property
    def iend(self):
        if self._inputs is None:
            raise ValueError('Trying to get iend, '+\
                'but inputs are not allocated. Please allocate')

        if self._iend is None:
            raise ValueError('Trying to get iend, '+\
                'but it is not set. Please set value')

        return self._iend


    @iend.setter
    def iend(self, value):
        ''' Set data '''
        value = np.int32(value)

        if self._inputs is None:
            raise ValueError('Trying to set iend, '+\
                'but inputs are not allocated. Please allocate')

        # Syntactic sugar to get a simulation running for the whole period
        if value == -1:
            value = self.ntimesteps-1

        if value<0 or value>self.ntimesteps-1:
            raise ValueError('Expected iend in [0, {0}], got {1}'.format(\
                self.ntimesteps-1, value))

        self._iend = value


    @property
    def inputs(self):
        if self._inputs is None:
            raise ValueError('Trying to access inputs, '+\
                'but they are not allocated. Please allocate')

        return self._inputs


    @inputs.setter
    def inputs(self, values):
        ''' Set data '''
        inputs = np.atleast_2d(values).astype(np.float64)
        if inputs.shape[1] != self.ninputs:
            raise ValueError('Expected {0} inputs, got {1}'.format(\
                self.ninputs, values.shape[1]))

        noutputs = max(1, self.noutputs)
        if self._inputs is None:
            self.allocate(inputs, noutputs)
        else:
            if self.ntimesteps != inputs.shape[1]:
                self.allocate(inputs, noutputs)
            else:
                self._inputs = values


    @property
    def noutputsmax(self):
        return self._noutputsmax


    @property
    def noutputs(self):
        return self._noutputs


    @property
    def outputs(self):
        if self._outputs is None:
            raise ValueError('Trying to access outputs, '+\
                'but they are not allocated. Please allocate')

        return self._outputs


    @outputs.setter
    def outputs(self, values):
        ''' Set data '''
        outputs = np.atleast_2d(values).astype(np.float64)
        noutputs = max(1, self.noutputs)
        if outputs.shape[1] != noutputs:
            raise ValueError('Expected {0} outputs, got {1}'.format(\
                noutputs, values.shape[1]))

        if self._inputs is None:
            inputs = np.zeros((outputs.shape[0], self.ninputs))
            self.allocate(inputs, noutputs)

        self._outputs = values


    def allocate(self, inputs, noutputs=1):
        ''' Allocate inputs and outputs arrays.
        We define the number of outputs here to allow more
        flexible memory allocation '''

        if noutputs <= 0 or noutputs > self.noutputsmax:
            raise ValueError(('model {0}: ' +\
                'Expected noutputs in [1, {1}], got {2}').format(\
                    self.name, self.noutputsmax, noutputs))

        # Allocate inputs
        inputs = np.atleast_2d(inputs)
        if inputs.shape[1] != self._ninputs:
            raise ValueError(('model {0}: Expected ninputs={1}, '+\
                                'got {2}').format(self.name, self._ninputs, \
                                inputs.shape[1]))
        self._inputs = inputs

        # Allocate outputs
        self._noutputs = noutputs
        self._outputs = np.zeros((inputs.shape[0], noutputs))

        # Set istart/iend to default
        self.istart = 0
        self.iend = -1


    def initialise(self, states=None, uhs=None):
        ''' Initialise state vector and potentially all UH states vectors '''
        if states is None:
            self.states.reset()
        else:
            self.states.values = states

        if not uhs is None:
            # Set uhs states values to argument
            nuh = self.params.nuh
            if len(uhs) != nuh:
                raise ValueError(('Expected a list of {0} unit'+\
                    ' hydrograpgs object for'+\
                    ' initialisation, got {1}').format(\
                    nuh, len(uhs)))

            for iuh in range(nuh):
                uh1 = self.params.uhs[iuh]
                uh2 = uhs[iuh]

                if uh1.nuh != uh2.nuh:
                    raise ValueError(('Expected nuh for UH[{0}] to be {1}.'+\
                            'Got {2}.').format(uh1.nuh, uh2.nuh))

                if abs(uh1.param-uh2.param)>1e-8:
                    raise ValueError(('Expected param for UH[{0}] to be {1}.'+\
                            'Got {2}.').format(uh1.param, uh2.param))

                uh1.reset()
                uh1.states[:uh1.nuh] = uh2.states[:uh2.nuh]

        else:
            # Reset uhs states values
            nuh = self.params.nuh
            for iuh in range(nuh):
                uh1 = self.params.uhs[iuh]
                uh1.reset()


    def run(self):
        ''' Run the model '''
        raise NotImplementedError(('model {0}: '+\
            'Method run not implemented').format(self.model))


    def run_checked(self):
        ''' Run the model following inputs, outputs and istart/iend checks '''

        if self._inputs is None:
            raise ValueError('Trying to run the model, '+\
                'but inputs are not allocated. Please allocate')

        if self._outputs is None:
            raise ValueError('Trying to run the model, '+\
                'but outputs are not allocated. Please allocate')

        if self.istart >= self.iend:
            raise ValueError('Expected istart({0}) < iend({1})'.format(\
                self.istart, self.iend))

        self.run()


    def clone(self):
        ''' Clone the current model instance'''
        return copy.deepcopy(self)


