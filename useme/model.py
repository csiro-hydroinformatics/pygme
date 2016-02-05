import math, itertools
import numpy as np

import c_hymod_models_utils

NUHMAXLENGTH = c_hymod_models_utils.uh_getnuhmaxlength()

class Vector(object):

    def __init__(self, id, nval, nens=1, prefix='X',
            has_minmax = True):

        self.id = id
        self._nval = nval
        self._nens = nens
        self.has_minmax = has_minmax
        self.prefix = prefix

        self._iens = 0

        self._data = np.nan * np.ones((nens, nval), dtype=np.float64)

        self._names = np.array(['{0}{1}'.format(prefix, i)
                                    for i in range(nval)])

        self._default = np.nan * np.ones(nval, dtype=np.float64)

        if has_minmax:
            self._min = -np.inf * np.ones(nval, dtype=np.float64)
            self._max = np.inf * np.ones(nval, dtype=np.float64)
            self._hitbounds = np.zeros(nens, dtype=np.bool)
        else:
            self._min = None
            self._max = None
            self._hitbounds = None

        self.means = 0 * np.ones(nval, dtype=np.float64)
        self.covar = np.eye(nval, dtype=np.float64)



    def __str__(self):

        str = 'Vector {0} : nval={1} nens={2} {{'.format( \
            self.id, self.nval, self.nens)
        str += ', '.join(self._names)
        str += '}'

        return str


    def __set_attrib(self, target, source):

        if target == '_weights' and not self.has_weights:
            raise ValueError(('Vector {0}: Cannot set weights, '
                'vector do not have this attribute').format(self.id))

        if target in ['_min', '_max'] and not self.has_minmax:
            raise ValueError(('Vector {0}: Cannot set min or max, '
                'vector do not have this attribute').format(self.id))

        _source = np.atleast_1d(source).flatten()

        if not target in ['_names']:
            _source = _source.astype(np.float64)

        if len(_source) != self.nval:
            raise ValueError(('Vector {0}: tried setting {1}, ' + \
                'got wrong size ({2} instead of {3})').format(\
                self.id, target, len(_source), self.nval))

        setattr(self, target, _source)


    def reset(self, value=None):
        ''' Set parameter data to a given value or default values if value is None '''
        nval = self.nval

        if value is None:
            value = self._default
        else:
            value = (value * np.ones(self.nval)).astype(np.float64)

        value = np.clip(value.flatten(), self.min, self.max)
        self._data = np.repeat(np.atleast_2d(value), self.nens, axis=0)


    def random(self, distribution='normal'):
        ''' Randomise vector data '''

        if not self.has_minmax and distribution=='uniform':
            raise ValueError(('Vector {0}: Cannot randomize with uniform ' +
                ' distribution if vector has no min/max').format(self.id))

        if (self.means is None or self.covar is None) and distribution=='uniform':
            raise ValueError(('Vector {0}: Cannot randomize with normal ' +
                ' distribution if vector has no means or covar').format(self.id))

        if distribution == 'normal':
            self._data = np.random.multivariate_normal(self.means,
                    self.covar, (self.nens, ))

        elif distribution == 'uniform':
            self._data = np.random.uniform(self.min, self.max,
                    (self.nens, self.nval))
        else:
            raise ValueError(('Vector {0}: Random, distribution ' +
                '{1} not allowed').format(self.id, distribution))


    @property
    def nens(self):
        return self._nens

    @property
    def nval(self):
        return self._nval

    @property
    def data(self):
        ''' Get data for a given ensemble member set by iens '''
        return self._data[self._iens, :]

    @data.setter
    def data(self, value):
        ''' Set data for a given ensemble member set by iens '''

        _value = np.atleast_1d(value).flatten()

        if len(_value) != self.nval:
            raise ValueError(('Vector {0} / ensemble {1}: tried setting data ' + \
                'with vector of wrong size ({2} instead of {3})').format(\
                self.id, self._iens, len(_value), self.nval))

        if self.has_minmax:
            # Avoids raising a warning with NaN substraction
            with np.errstate(invalid='ignore'):
                hitb = np.subtract(_value, self._min) < 0.
                hitb = hitb | (np.subtract(self._max, _value) < 0.)

            self._hitbounds[self._iens] = np.any(hitb)
            self._data[self._iens, :] = np.clip(_value, self._min, self._max)

        else:
            self._data[self._iens, :] = _value


    @property
    def min(self):
        return self._min

    @min.setter
    def min(self, value):
        self.__set_attrib('_min', value)



    @property
    def max(self):
        return self._max

    @max.setter
    def max(self, value):
        self.__set_attrib('_max', value)


    @property
    def default(self):
        return self._default

    @default.setter
    def default(self, value):
        self.__set_attrib('_default', value)

        if self.has_minmax:
            self._default = np.clip(self._default, self._min, self._max)


    @property
    def names(self):
        return self._names

    @names.setter
    def names(self, value):
        self.__set_attrib('_names', value)


    @property
    def hitbounds(self):
        return self._hitbounds[self._iens]


    @property
    def iens(self):
        return self._iens

    @iens.setter
    def iens(self, value):
        ''' Set the ensemble number. Checks that number is not greater that total number of ensembles '''

        value = int(value)
        if value >= self.nens or value < 0:
            raise ValueError(('Vector {0}: iens {1} ' \
                    '>= nens {2} or < 0').format( \
                                self.id, value, self.nens))

        self._iens = value


    def clone(self):
        clone = Vector(self.id, self.nval, self.nens,
                    self.prefix, self.has_minmax)

        clone.iens = self.iens
        clone.names = self.names.copy()

        clone._data = self._data.copy()

        if self.has_minmax:
            clone.min = self.min.copy()
            clone.max = self.max.copy()
            clone._hitbounds = [h.copy() for h in self._hitbounds]

        clone.means = self.means.copy()
        clone.covar = self.covar.copy()

        return clone




class Matrix(object):

    def __init__(self, id, nval, nvar, nens=1, data=None, prefix='V'):
        self.id = id
        self.prefix = prefix
        self._iens = 0

        if nval is not None and nvar is not None and nens is not None:
            self._nval = nval
            self._nvar = nvar
            self._nens = nens
            self._data = np.nan * np.ones((nens, nval, nvar), dtype=np.float64)

        elif data is not None:

            _data = data
            if len(data.shape) == 1:
                _data = np.atleast_3d(data)
            elif len(data.shape) == 2:
                _data = np.atleast_3d(data.T).T

            self._nens, self._nval, self._nvar = _data.shape
            self._data = _data

        else:
            raise ValueError(('{0} matrix, ' + \
                    'Wrong arguments to Matrix.__init__').format(self.id))

        self._names = ['{0}{1}'.format(prefix, i) for i in range(self.nvar)]


    @classmethod
    def fromdims(cls, id, nval, nvar, nens=1, prefix='V'):
        return cls(id, nval, nvar, nens, None)


    @classmethod
    def fromdata(cls, id, data):
        return cls(id, None, None, None, data, prefix='V')

    def __str__(self):
        str = 'Matrix {0} : nval={1} nvar={2} nens={3} {{'.format( \
            self.id, self.nval, self.nvar, self.nens)
        str += ', '.join(self._names)
        str += '}'

        return str


    @property
    def nvar(self):
        return self._nvar

    @property
    def nval(self):
        return self._nval

    @property
    def nens(self):
        return self._nens


    @property
    def names(self):
        return self._names

    @names.setter
    def names(self, value):
        self._names = np.atleast_1d(value)

        if len(self._names) != self.nvar:
            raise ValueError(('{0} matrix: tried setting _names, ' + \
                'got wrong size ({1} instead of {2})').format(\
                self.id, len(self._names), self.nvar))


    @property
    def data(self):
        return self._data[self._iens]

    @data.setter
    def data(self, value):
        _value = np.atleast_2d(value)

        if _value.shape[0] == 1:
            _value = _value.T

        if _value.shape[0] != self.nval:
            raise ValueError(('{0} matrix: tried setting _data,' + \
                    ' got wrong number of values ' + \
                    '({1} instead of {2})').format( \
                    self.id, _value.shape[0], self.nval))

        if _value.shape[1] != self.nvar:
            raise ValueError(('{0} matrix: tried setting _data,' + \
                    ' got wrong number of variables ' + \
                    '({1} instead of {2})').format( \
                    self.id, _value.shape[1], self.nvar))

        self._data[self._iens, :, :] = _value


    @property
    def iens(self):
        return self._iens

    @iens.setter
    def iens(self, value):
        ''' Set the ensemble number. Checks that number is not greater that total number of ensembles '''

        value = int(value)
        if value >= self.nens or value < 0:
            raise ValueError(('Vector {0}: iens {1} ' \
                    '>= nens {2} or < 0').format( \
                                self.id, value, self.nens))

        self._iens = value


    def reset(self, value=0.):
        nens = self.nens
        nval = self.nval
        nvar = self.nvar

        self._data = value * np.ones((nens, nval, nvar), dtype=np.float64)



class Model(object):

    def __init__(self, name,
            nconfig,
            ninputs,
            nparams,
            nstates,
            noutputs_max,
            nens_params=1,
            nens_states_random=1,
            nens_outputs_random=1):

        self.name = name
        self.ninputs = ninputs

        self.nuhlength = 0
        self.noutputs_max = noutputs_max

        # Random ensemble
        self._iens_outputs_random = 0
        self.nens_outputs_random = nens_outputs_random

        self._iens_states_random = 0
        self.nens_states_random = nens_states_random

        # Config vector
        self.config = Vector('config', nconfig)

        # Param vector
        self._params = Vector('params', nparams, nens_params, prefix='X')

        # UH ordinates. Number of ensembles is same than nens_params
        self._uh = Vector('uh', NUHMAXLENGTH, nens_params)

        # Initialize UH to [1, 0, 0, .., 0] (neutral UH)
        for iens in range(nens_params):
            self._uh.iens = iens
            self._uh.data = [1.] + [0.] * (NUHMAXLENGTH-1)

        self.nstates = nstates
        self._states = None
        self._statesuh = None

        self._inputs = None
        self._outputs = None

        self.idx_start = 0
        self.idx_end = 0


    def __str__(self):
        str = '\n{0} model implementation\n'.format( \
            self.name)

        nens = 0
        ninputs = 0
        if not self._inputs is None:
            ninputs = self._inputs.nvar
            nens = self._inputs.nens
        str += '  ninputs      = {0} [{1} ens]\n'.format(ninputs, nens)

        str += '  nparams      = {0} [{1} ens]\n'.format(
                    self._params.nval, self._params.nens)

        if not self._states is None:
            str += '  nstates      = {0} [{1} ens]\n'.format(
                    self._states.nval, self.nens_states_random)

        str += '  noutputs = {0} (max) [{1} ens]\n'.format(self.noutputs_max,
                    self.nens_outputs_random)

        str += '  nuhlengthmax    = {0}\n'.format(self._uh.nval)

        return str


    @property
    def inputs(self):
        return self._inputs.data

    @inputs.setter
    def inputs(self, value):
        self._inputs.data = value


    @property
    def iens_inputs(self):
        return self._inputs.iens

    @iens_inputs.setter
    def iens_inputs(self, value):
        self._inputs.iens = value


    @property
    def params(self):
        return self._params.data


    @params.setter
    def params(self, value):
        self._params.data = value

        # When setting params, check uh are in sync
        self._uh.iens = self._params.iens
        self.set_uh()


    @property
    def iens_params(self):
        return self._params.iens

    @iens_params.setter
    def iens_params(self, value):
        ''' Set the parameter parameter ensemble number '''
        self._params.iens = value
        self._uh.iens = value


    @property
    def uh(self):
        return self._uh.data

    @uh.setter
    def uh(self, value):
        ''' Set UH values '''
        self._uh.data = value


    @property
    def states(self):
        # Code needed to sync input/params ensemble with states
        self.iens_states_random = self.iens_states_random
        return self._states.data

    @states.setter
    def states(self, value):
        self._states.data = value


    @property
    def statesuh(self):
        # Code needed to sync input/params ensemble with states
        self.iens_states_random = self.iens_states_random
        return self._statesuh.data

    @statesuh.setter
    def statesuh(self, value):
        self._statesuh.data = value


    @property
    def iens_states_random(self):
        return self._iens_states_random

    @iens_states_random.setter
    def iens_states_random(self, value):
        ''' Set the states vector ensemble number '''

        value = int(value)
        if value >= self.nens_states_random or value < 0:
            raise ValueError(('Model {0}: iens_states_random {1} ' \
                    '>= nens {2} or < 0').format( \
                                self.name, value, self.nens_states_random))

        # Determines the state ensemble number given the input ensemble and the parameter ensemble
        # It is assumed that the loop will go first on inputs then on parameters
        # and finally on random states ensembles
        n1 = self._params.nens
        n2 = self.nens_states_random
        iens = n1 * n2  * self._inputs.iens
        iens += n2 * self._params.iens
        iens += value

        # Set ensemble number
        self._iens_states_random = value
        self._states.iens = iens
        self._statesuh.iens = iens


    @property
    def outputs(self):
        return self._outputs.data

    @outputs.setter
    def outputs(self, value):
        self._outputs.data = value


    @property
    def iens_outputs_random(self):
        return self._iens_outputs_random

    @iens_outputs_random.setter
    def iens_outputs_random(self, value):
        ''' Set the outputs vector ensemble number '''

        if value >= self.nens_outputs_random or value < 0:
            raise ValueError(('Model {0}: iens_outputs_random {1} ' \
                    '>= nens {2} or < 0').format( \
                                self.name, value, self.nens_outputs_random))

        # Determines the state ensemble number given the input ensemble and the parameter ensemble
        # It is assumed that the loop will go first on inputs then on parameters
        # and finally on random states ensembles
        n1 = self._params.nens
        n2 = self.nens_states_random
        n3 = self.nens_outputs_random
        iens = n1 * n2  * n3 * self._inputs.iens
        iens += n2 * n3 * self._params.iens
        iens += n3 * self._iens_states_random
        iens += value

        # Set ensemble number
        self._outputs_states_random = value


    def getdims(self):
        nens = 0
        nval = 0
        nvar = 0
        if not self._inputs is None:
            nvar = self._inputs.nvar
            nval = self._inputs.nval
            nens = self._inputs.nens
        dims = {'inputs':{'nens':nens, 'nval':nval, 'nvar':nvar}}

        nens = self._params.nens
        nval = self._params.nval
        dims['params'] = {'nens':nens, 'nval':nval,
            'nuhlengthmax':self._uh.nval}

        nens = 0
        nval = 0
        if not self._states is None:
            nens = self._states.nens
            nval = self._states.nval
        dims['states'] = {'nens':nens, 'nval':nval,
            'nens_states_random':self.nens_states_random}

        nens = 0
        nval = 0
        nvar = 0
        if not self._outputs is None:
            nvar = self._outputs.nvar
            nval = self._outputs.nval
            nens = self._outputs.nens
        dims['outputs'] = {'nens':nens, 'nval':nval, 'nvar':nvar,
            'noutputs_max':self.noutputs_max,
            'nens_outputs_random':self.nens_outputs_random}

        return dims


    def allocate(self, nval, noutputs=1, nens_inputs=1):
        ''' We define the number of outputs here to allow more flexible memory allocation '''

        if noutputs <= 0:
            raise ValueError(('Number of outputs defined' + \
                ' for model {0} should be >0').format(nval))

        if noutputs > self.noutputs_max:
            raise ValueError(('Too many outputs defined for model {0}:' + \
                ' noutputs({1}) > noutputs_max({2})').format( \
                self.name, noutputs, self.noutputs_max))

        self._inputs = Matrix.fromdims('inputs',
                nval, self.ninputs, nens_inputs, prefix='I')

        # Allocate state vectors with number of ensemble
        nens = self._params.nens * self._inputs.nens * self.nens_states_random
        self._states = Vector('states', self.nstates, nens,
                                prefix='S', has_minmax=False)

        self._statesuh = Vector('statesuh', NUHMAXLENGTH, nens,
                                prefix='SUH', has_minmax=False)

        # Allocate output matrix with number of final ensemble
        nens *= self.nens_outputs_random
        self._outputs = Matrix.fromdims('outputs',
                nval, noutputs, nens, prefix='O')


    def set_uh(self):
        pass


    def initialise(self, states=None, statesuh=None):

        for iens in range(self._states.nens):
            if states is None:
                self._states.iens = iens
                self._states.data = [0.] * self._states.nval

            if statesuh is None:
                self._states.iens = iens
                self._statesuh.data = [0.] * self._statesuh.nval


    def run(self):
        pass

    def run_ens(self,
            ens_inputs = None,
            ens_params = None,
            ens_states_random = None,
            ens_outputs_random = None):
        ''' Run the model with selected ensembles '''

        # Set ensemble lists
        if ens_inputs is None:
            ens_inputs = range(self._inputs.nens)

        if ens_params is None:
            ens_params = range(self._params.nens)

        if ens_states_random is None:
            ens_states_random = range(self.nens_states_random)

        if ens_outputs_random is None:
            ens_outputs_random = range(self.nens_outputs_random)

        # Loop through ensembles
        for i_inputs, i_params, i_states, i_outputs in \
                itertools.product(ens_inputs, ens_params,
                        ens_states_random, ens_outputs_random):

            self.iens_inputs = i_inputs
            self.iens_params = i_params
            self.iens_states_random = i_states
            self.iens_outputs_random = i_outputs

            self.run()


    def clone(self):
        ''' Clone the current object model '''

        model = Model(self.name,
            self.config.nval,
            self.ninputs,
            self._params.nval,
            self.nstates,
            self.noutputs_max,
            self._params.nens,
            self.nens_states_random,
            self.nens_outputs_random)

        model.config.data = self.config.data.copy()

        for iens in range(self._params.nens):
            self._params.iens = iens
            model._params.iens = iens
            model._params.data = self._params.data.copy()


        if not self._inputs is None:
            model.allocate(self._inputs.nval, self._outputs.nvar,
                    self._inputs.nens)

            for iens in range(self._states.nens):
                self._states.iens = iens
                model._states.iens = iens
                model._states.data = self._states.data.copy()

                model._statesuh.iens = iens
                model._statesuh.data = self._statesuh.data.copy()

            for iens in range(self._inputs.nens):
                self._inputs.iens = iens
                model._inputs.iens = iens
                model._inputs.data = self._inputs.data.copy()

            for iens in range(self._outputs.nens):
                self._outputs.iens = iens
                model._outputs.iens = iens
                model._outputs.data = self._outputs.data.copy()

        return model


