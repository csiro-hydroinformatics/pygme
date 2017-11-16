import math, itertools
import copy
import random
import numpy as np

import c_pygme_models_utils

NORDMAXMAX = c_pygme_models_utils.uh_getnuhmaxlength()

from hydrodiy.data.containers import Vector


UHNAMES = ['gr4j_ss1_daily', 'gr4j_ss2_daily', \
            'gr4j_ss1_hourly', 'gr4j_ss2_hourly', \
            'lag', 'triangle', 'flat']

class UH(object):

    def __init__(self, name, nordmax=NORDMAXMAX):
        ''' Object handling unit hydrograph. The object does not run the
        convolution, just stores the unit hydrograph ordinates

        Parameters
        -----------
        name : str
            Name of the UH
        nordmax : int
            Maximum number of ordinates

        '''

        # Set max length of uh
        if nordmax > NORDMAXMAX or nordmax<=0:
            raise ValueError(('Expected nuhmax in [1, {0}], '+\
                    'got {1}').format(NORDMAXMAX, nordmax))

        self._nordmax = nordmax

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
        self._ord = np.zeros(nordmax, dtype=np.float64)
        self._ord[0] = 1.
        self._states = np.zeros(nordmax, dtype=np.float64)

        # set time base param value
        self._timebase = 0.

        # Number of ordinates
        # nuh is stored as an array to be passed to the C routine
        self._nord = np.array([1], dtype=np.int32)


    def __str__(self):
        str = 'UH {0}: timebase={1} nord={2}'.format(self.name, \
                self.timebase, self.nord)
        return str


    @property
    def uhid(self):
        return self._uhid


    @property
    def nordmax(self):
        return self._nordmax


    @property
    def nord(self):
        # There is a trick here, we return an
        # integer but the internal state is an array
        return self._nord[0]


    @property
    def timebase(self):
        return self._timebase

    @timebase.setter
    def timebase(self, value):
        # Check value
        value = np.float64(value)

        # Populate the uh ordinates
        ierr = c_pygme_models_utils.uh_getuh(self.nordmax, self.uhid, \
                                        value, self._nord, self._ord)
        if ierr>0:
            raise ValueError(('When setting param to {0} for UH {1}, '+\
                'c_pygme_models_utils.uh_getuh returns {2}').format(\
                        value, self.name, ierr))

        # Store parameter value
        self._timebase = value

        # Reset uh states to a vector of zeros
        # with length nord
        self._states[:self.nord] = 0

        # Set remaining ordinates to 0
        self._ord[self.nord:] = 0


    @property
    def ord(self):
        return self._ord


    @property
    def states(self):
        return self._states

    @states.setter
    def states(self, values):
        values = np.atleast_1d(values).astype(np.float64)

        nord = self.nord
        if values.shape[0]<nord:
            raise ValueError('Expected state vector to be of length'+\
                '>={0}, got {1}'.format(nord, values.shape[0]))

        self.reset()
        self._states[:nord] = values[:nord]


    def reset(self):
        self._states = np.zeros(self.nordmax)


    def clone(self):
        clone = UH(self.name, self.nordmax)
        clone.timebase = self.timebase
        clone._states = self.states.copy()

        return clone



# Overload Vector class to change UH and corresponding states
# when changing model parameters
class ParamsVector(Vector):

    def __init__(self, params):
        ''' Object handling parameter vector. The object stores the unit
        hydrographs and the functions used to set the uh time base

        Parameters
        -----------
        params : hydrodiy.data.containers.Vector
            Vector of parameters including names, default values, min and max.
        uhs : list
            list of tuples containing for each unit hydrograph a function to
            setup the time base from parameter values and a UH object.

        Example
        -----------
        This code produces a parameter vector with two uh attached.
        The first uh is controled by the first parameter (X1),
        the second uh is controled by the expression X2+X3/2

        >>> params = Vector(['X1', 'X2', 'X3'])
        >>> uhs =[(lambda params: params.X1, UH("lag")), (lambda params: params.X1+params.X1/2, UH("lag")]
        >>> pv = ParamsVector(params, uhs)

        '''
        # check_hitbounds is turned on
        super(ParamsVector, self).__init__(params.names, \
                    params.defaults, params.mins, params.maxs, \
                    True)

        self._uhs = None


    def __setattr__(self, name, value):
        # Set attribute for vector object
        super(ParamsVector, self).__setattr__(name, value)

        # Set UH parameter if needed
        if not hasattr(self, 'names'):
            return

        if name in self.names:
            if self.nuh>0:
                for iuh, (set_timebase, uh) in enumerate(self.uhs):
                    uh.timebase = set_timebase(self)


    @property
    def nuh(self):
        if self.uhs is None:
            return 0
        else:
            return len(self.uhs)


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

        # Set UH parameter if needed
        if self.nuh>0:
            for iuh, (set_timebase, uh) in enumerate(self.uhs):
                uh.timebase = set_timebase(self)


    def add_uh(self, uh_name, set_timebase, nuhmax=NORDMAXMAX):
        ''' Add uh object '''

        if self._uhs is None:
            self._uhs = []

        test = set_timebase(self)
        if not isinstance(test, float):
            raise ValueError(('Expected set_timebase function to '+\
                'return a float, got {0}').format(test))

        # Create UH
        uh = UH(uh_name, nuhmax)

        # Set timebase to check it does not trigger any error
        uh.timebase = test

        # All test ok, appending uh to list of uhs
        self._uhs.append((set_timebase, uh))


    def clone(self):
        params = Vector(self.names, self.defaults, self.mins, \
                    self.maxs, self.check_hitbounds)

        clone = ParamsVector(params)
        clone.values = self.values.copy()
        if not self.uhs is None:
            clone._uhs = [(uht[0], uht[1].clone()) for uht in self.uhs]

        return clone


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


    def __getattribute__(self, name):
        # Except certain names to avoid infinite recursion
        if name in ['name', '_config', '_params', '_states', '_ninputs', \
            '_noutputsmax', '_noutputs', '_inputs', '_outputs', '_istart', \
            '_iend']:
            return super(Model, self).__getattribute__(name)

        if name in self._params.names:
            return getattr(self._params, name)

        if name in self._config.names:
            return getattr(self._config, name)

        if name in self._states.names:
            return getattr(self._states, name)

        return super(Model, self).__getattribute__(name)


    def __setattr__(self, name, value):
        # Except certain names to avoid infinite recursion
        if name in ['name', '_config', '_params', '_states', '_ninputs', \
            '_noutputsmax', '_noutputs', '_inputs', '_outputs', '_istart', \
            '_iend']:
            super(Model, self).__setattr__(name, value)
            return

        if name in self._params.names:
            return setattr(self._params, name, value)

        elif name in self._config.names:
            return setattr(self._config, name, value)

        elif name in self._states.names:
            return setattr(self._states, name, value)

        else:
            super(Model, self).__setattr__(name, value)


    def __str__(self):
        str = ('\n{0} model implementation\n'+\
                '\tConfig: {1}\n\tParams: {2}\n\tStates: {3}'+\
                '\n\tNUH: {4}').format( \
                    self.name, self.config, self.params, \
                    self.states, self.params.nuh)
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
            raise ValueError(('model {0}: Trying to set iend, '+\
                'but inputs are not allocated. Please allocate').format(self.name))

        # Syntactic sugar to get a simulation running for the whole period
        if value == -1:
            value = self.ntimesteps-1

        if value<0 or value>self.ntimesteps-1:
            raise ValueError('model {0}: Expected iend in [0, {1}], got {2}'.format(\
                self.name, self.ntimesteps-1, value))

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
        inputs = np.ascontiguousarray(np.atleast_2d(values).astype(np.float64))
        if inputs.shape[1] != self.ninputs:
            raise ValueError('model {0}: Expected {1} inputs, got {2}'.format(\
                self.name, self.ninputs, values.shape[1]))

        self._inputs = inputs


    @property
    def noutputsmax(self):
        return self._noutputsmax


    @property
    def noutputs(self):
        return self._noutputs


    @property
    def outputs(self):
        if self._outputs is None:
            raise ValueError(('model {0}: Trying to access outputs, '+\
                'but they are not allocated. Please allocate').format(\
                    self.name))

        return self._outputs


    @outputs.setter
    def outputs(self, values):
        ''' Set data '''
        outputs = np.ascontiguousarray(np.atleast_2d(values).astype(np.float64))
        noutputs = max(1, self.noutputs)

        if outputs.shape[1] != noutputs:
            raise ValueError('model {0}: Expected {1} outputs, got {2}'.format(\
                self.name, noutputs, values.shape[1]))

        self._outputs = outputs


    def allocate(self, inputs, noutputs=1):
        ''' Allocate inputs and outputs arrays.
        We define the number of outputs here to allow more
        flexible memory allocation '''

        if noutputs <= 0 or noutputs > self.noutputsmax:
            raise ValueError(('model {0}: ' +\
                'Expected noutputs in [1, {1}], got {2}').format(\
                    self.name, self.noutputsmax, noutputs))

        # Allocate inputs
        self.inputs = inputs

        # Allocate outputs
        self._noutputs = noutputs
        self.outputs = np.zeros((inputs.shape[0], noutputs))

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
                    ' hydrographs object for'+\
                    ' initialisation, got {1}').format(\
                    nuh, len(uhs)))

            for iuh in range(nuh):
                # We extract the UH object
                # the set_timebase function is not needed here
                _, uh1 = self.params.uhs[iuh]

                # Compare with the uh supplied to initialise
                uh2 = uhs[iuh]

                if uh1.nord != uh2.nord:
                    raise ValueError(('Expected UH[{0}] nord to be {1}.'+\
                            ' Got {2}.').format(uh1.nord, uh2.nord))

                if abs(uh1.timebase-uh2.timebase)>1e-8:
                    raise ValueError(('Expected UH[{0}] timebase to be {1}.'+\
                            ' Got {2}.').format(iuh, uh1.timebase, uh2.timebase))

                uh1.reset()
                uh1.states[:uh1.nord] = uh2.states[:uh2.nord]

        else:
            # Reset uhs states values
            nuh = self.params.nuh
            for iuh in range(nuh):
                # We extract the UH object
                # the set_timebase function is not needed here
                _, uh1 = self.params.uhs[iuh]
                uh1.reset()


    def run(self):
        ''' Run the model '''
        raise NotImplementedError(('model {0}: '+\
            'Method run not implemented').format(self.name))


    def clone(self):
        ''' Clone the current model instance'''

        model = Model(self.name, \
            self.config.clone(), \
            self.params.clone(), \
            self.states.clone(), \
            self.ninputs, \
            self.noutputsmax)

        # Allocate data
        if not self._inputs is None:
            model.allocate(self.inputs, self.noutputs)
            model.istart = self.istart
            model.iend = self.iend

        return model


