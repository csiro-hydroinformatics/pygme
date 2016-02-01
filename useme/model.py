import math
import numpy as np

import c_hymod_models_utils

NUHMAXLENGTH = c_hymod_models_utils.uh_getnuhmaxlength()


class Vector(object):

    def __init__(self, id, nval, nens=1):
        self._id = id
        self._nval = nval
        self._nens = nens
        self._iens = 0

        self._data = [np.nan * np.ones(nval).astype(np.float64)] * nens

        self._names = np.array(['X{0}'.format(i) for i in range(nval)])
        self._units = ['-'] * nval
        self._min = -np.inf * np.ones(nval).astype(np.float64)
        self._max = np.inf * np.ones(nval).astype(np.float64)
        self._default = np.nan * np.ones(nval).astype(np.float64)

        self._hitbounds = [False] * nens

        self._setter_decorator = None
        self._iens_decorator = None


    def __str__(self):

        str = 'Vector {0} : nval={1} nens={2} {{'.format( \
            self._id, self._nval, self._nens)
        str += ', '.join(self._names)
        str += '}'

        return str

    def __checkname(self, name):

        try:
            kx = np.where(name == self._names)[0]
        except ValueError:
            raise ValueError(('Vector {0}: name {1} not in' + \
                    ' the vector names').format(self._id, name))
        return kx


    def __set_attrib(self, target, source):

        _source = np.atleast_1d(source).flatten()

        if not target in ['_names', '_units']:
            _source = _source.astype(np.float64)

        if len(_source) != self.nval:
            raise ValueError(('Vector {0}: tried setting {1}, ' + \
                'got wrong size ({2} instead of {3})').format(\
                self._id, target, len(_source), self._nval))

        setattr(self, target, _source)


    def __getitem__(self, name):

        kx = self.__checkname(name)

        return self._data[self._iens][kx]


    def __setitem__(self, name, value):

        kx = self.__checkname(name)

        data = self._data[self._iens]
        data[kx] = value


    def reset(self, value=None, iens=None):
        nens = self._nens
        nval = self._nval

        if value is None:
            default = self._default
        else:
            default = np.array([value] * nval).astype(np.float64)

        if iens is None:
            self._data = [default.copy() for i in range(nens)]
        else:
            self._data[iens] = default.copy()


    @property
    def nval(self):
        return self._nval


    @property
    def nens(self):
        return self._nens


    @property
    def data(self):
        ''' Get data for a given ensemble member set by iens '''
        return self._data[self._iens]

    @data.setter
    def data(self, value):
        ''' Set data for a given ensemble member set by iens '''

        _value = np.atleast_1d(value).flatten()

        if len(_value) != self.nval:
            raise ValueError(('Vector {0} / ensemble {1}: tried setting data ' + \
                'with vector of wrong size ({2} instead of {3})').format(\
                self._id, self._iens, len(_value), self._nval))

        hitb = np.subtract(_value, self._min) < 0.
        hitb = hitb | (np.subtract(self._max, _value) < 0.)
        self._hitbounds[self._iens] = np.any(hitb)

        self._data[self._iens] = np.clip(_value, self._min, self._max)

        # Run all methods from the post setter object
        # (e.g. to change UH ordinates when a parameter vector changes)
        if not self._setter_decorator is None:
            self._setter_decorator.run()


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
        self._default = np.clip(self._default, self._min, self._max)


    @property
    def names(self):
        return self._names

    @names.setter
    def names(self, value):
        self.__set_attrib('_names', value)


    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, value):
        self.__set_attrib('_units', value)


    @property
    def hitbounds(self):
        return self._hitbounds[self._iens]


    @property
    def iens(self):
        return self._iens

    @iens.setter
    def iens(self, value):
        self.set_iens(value)

    def set_iens(self, value):
        ''' Set the ensemble number. Checks that number is not greater that total number of ensembles '''
        if value >= self._nens or value < 0:
            raise ValueError(('Vector {0}: iens {1} ' \
                    '>= nens {2} or < 0').format( \
                                self._id, value, self._nens))

        self._iens = value

        if not self._iens_decorator is None:
            self._iens_decorator.run()


class VectorDecorator(object):
    ''' Object used to decorate a vector with method from external objects '''

    def __init__(self, vector):
        self.vector = vector
        self.args = {}


    def add_method(self, obj, method):
        ''' Add a method from an external object to the decorator '''
        self.args[(obj, method)] = None


    def set_args(self, obj, method, args):
        ''' Set argument for the method. Caution: args should a dict, e.g. (3.4, )'''

        key = (obj, method)
        if not key in self.args:
            raise ValueError(('Key (Object {0}, Method {1}) not in the list' +
                ' of defined methods').format(obj, method))
        else:
            self.args[key] = args


    def run(self):
        ''' Run all defined methods with arguments '''

        for key in self.args:
            # get method
            obj = key[0]
            methodname = key[1]
            method = getattr(obj, methodname)

            # Run method
            args = self.args[key]
            if args is None:
                method()
            else:
                method(*args)



class Matrix(object):

    def __init__(self, id, nval, nvar, nens=1, data=None):
        self._id = id
        self._iens = 0

        if nval is not None and nvar is not None and nens is not None:
            self._nval = nval
            self._nvar = nvar
            self._nens = nens
            self._data = [np.nan * np.ones((nval, nvar)).astype(np.float64)]*nens

        elif data is not None:

            if isinstance(data, np.ndarray):
                data = [data]

            if not isinstance(data, list):
                raise ValueError('data is not a list or a numpy.ndarray')

            self._nens = len(data)

            _data = []
            for idx, d in enumerate(data):
                _d = np.atleast_2d(d)
                if _d.shape[0] == 1:
                    _d = _d.T
                _data.append(_d)

                if idx == 0:
                    self._nval, self._nvar = _d.shape
                else:
                    if _d.shape != (self._nval, self._nvar):
                        raise ValueError(('Shape of element {0} in data ({1}) ' + \
                            ' is different from ({2}, {3})'.format( \
                            idx,  _d.shape, self._nval, self._nvar)))

            self._data = _data

        else:
            raise ValueError(('{0} matrix, ' + \
                    'Wrong arguments to Matrix.__init__').format(self._id))

        self._names = ['V{0}'.format(i) for i in range(self._nvar)]


    @classmethod
    def fromdims(cls, id, nval, nvar, nens=1):
        return cls(id, nval, nvar, nens, None)


    @classmethod
    def fromdata(cls, id, data):
        return cls(id, None, None, None, data)

    def __str__(self):
        str = 'Matrix {0} : nval={1} nvar={2} nens={3} {{'.format( \
            self._id, self._nval, self._nvar, self._nens)
        str += ', '.join(self._names)
        str += '}'

        return str


    @property
    def nval(self):
        return self._nval


    @property
    def nens(self):
        return self._nens

    @property
    def nvar(self):
        return self._nvar


    @property
    def names(self):
        return self._names

    @names.setter
    def names(self, value):
        self._names = np.atleast_1d(value)

        if len(self._names) != self._nvar:
            raise ValueError(('{0} matrix: tried setting _names, ' + \
                'got wrong size ({1} instead of {2})').format(\
                self._id, len(self._names), self._nvar))


    @property
    def data(self):
        return self._data[self._iens]

    @data.setter
    def data(self, value):
        _value = np.atleast_2d(value)

        if _value.shape[0] == 1:
            _value = _value.T

        if _value.shape[0] != self._nval:
            raise ValueError(('{0} matrix: tried setting _data,' + \
                    ' got wrong number of values ' + \
                    '({1} instead of {2})').format( \
                    self._id, _value.shape[0], self._nval))

        if _value.shape[1] != self._nvar:
            raise ValueError(('{0} matrix: tried setting _data,' + \
                    ' got wrong number of variables ' + \
                    '({1} instead of {2})').format( \
                    self._id, _value.shape[1], self._nvar))

        self._data[self._iens] = _value


    @property
    def iens(self):
        return self._iens

    @iens.setter
    def iens(self, value):
        ''' Set the ensemble number. Checks that number is not greater that total number of ensembles '''
        if value >= self._nens or value < 0:
            raise ValueError(('Vector {0}: iens {1} ' \
                    '>= nens {2} or < 0').format( \
                                self._id, value, self._nens))

        self._iens = value


    def reset(self, value=0., iens=None):
        nens = self._nens
        nval = self._nval
        nvar = self._nvar

        default = (value * np.ones((nval, nvar))).astype(np.float64)

        if iens is None:
            self._data = [default.copy() for i in range(nens)]
        else:
            self._data[iens] = default.copy()



class Model(object):

    def __init__(self, name,
            nconfig,
            ninputs,
            nparams,
            nstates,
            noutputs_max,
            inputs_names,
            outputs_names,
            nens_params=1,
            nens_states_generated=1,
            nens_outputs_generated=1):

        self._name = name
        self._ninputs = ninputs

        self._nuhlength = 0
        self._noutputs_max = noutputs_max
        self._nens_outputs = nens_outputs

        self._nens_states_generated = nens_states_generated

        self._config = Vector('config', nconfig)
        self._params = Vector('params', nparams, nens_params)
        self._params_default = Vector('params_default', nparams)

        # UH ordinates. Number of ensembles is same than nens_params
        self._uh = Vector('uh', NUHMAXLENGTH, nens_params)

        # Initialize UH to [1, 0, 0, .., 0] (neutral UH)
        for iens in range(nens_params):
            self._uh.iens = iens
            self._uh.data = [1.] + [0.] * (NUHMAXLENGTH-1)

        # This code is used to change UH ordinates when
        # model parameters are changed
        setter_dec = VectorDecorator(self._params)
        setter_dec.add_method(self, 'set_uh')
        self._params._setter_decorator = setter_dec

        # This code is used to link ensemble number between parameters
        # and UH
        iens_dec = VectorDecorator(self._params)
        iens_dec.add_method(self._uh, 'set_iens')
        iens_dec.set_args(self._uh, 'set_iens', self._params.iens)
        # DOES NOT WORK !!!!
        self._params._iens_decorator = iens_dec

        # Number of states depends on number of inputs and parameters
        nens_states_final = nens_states * nens_inputs * nens_params
        self._states = Vector('states', nstates, nens_states_final)

        self._statesuh = Vector('statesuh', NUHMAXLENGTH, nens_states_final)

        self._inputs = None
        self._outputs = None

        if len(inputs_names) != ninputs:
            raise ValueError(('Model {0}: len(inputs_names)({1}) != ' + \
                    'ninputs ({2})').format(self._name, \
                        len(inputs_names), ninputs))

        self._inputs_names = inputs_names

        if len(outputs_names) != noutputs_max:
            raise ValueError(('Model {0}: len(outputs_names)({1}) != ' + \
                    'noutputs_max ({2})').format(self._name, \
                        len(outputs_names), \
                        noutputs_max))

        self._outputs_names = outputs_names


    def __str__(self):
        str = '\n{0} model implementation\n'.format( \
            self._name)
        str += '  ninputs      = {0} [{1} ens]\n'.format(
                    self._ninputs, self.inputs.nens)
        str += '  nparams      = {0} [{1} ens]\n'.format(
                    self.params.nval, self.params.nens)
        str += '  nstates      = {0} [{1} ens]\n'.format(
                    self.states.nval, self.states.nens)
        str += '  nuhmaxlength = {0}\n'.format(self._statesuh.nval)
        str += '  nuhlength    = {0} [{1} ens]\n'.format(self._nuhlength)

        return str


    @property
    def name(self):
        return self._name


    @property
    def config(self):
        return self._config


    @property
    def uh(self):
        return self._uh


    @property
    def statesuh(self):
        return self._statesuh


    @property
    def states(self):
        return self._states


    @property
    def params(self):
        return self._params


    @property
    def inputs(self):
        return self._inputs


    @property
    def outputs(self):
        return self._outputs

    @property
    def iens_params(self):
        return self._params.iens

    @iens_params.setter
    def iens_params(self, value):
        ''' Set the parameter ensemble number '''
        self._params.iens = value
        self._uh.iens = value


    def allocate(self, nval, noutputs=1):
        ''' We define the number of outputs here to allow more flexible memory allocation '''

        if noutputs <= 0:
            raise ValueError(('Number of outputs defined' + \
                ' for model {0} should be >0').format(nval))

        if noutputs > self._noutputs_max:
            raise ValueError(('Too many outputs defined for model {0}:' + \
                ' noutputs({1}) > noutputs_max({2})').format( \
                self._name, noutputs, self._noutputs_max))

        self._inputs = Matrix.fromdims('inputs',
                nval, self._ninputs, self._nens_inputs)

        self._inputs.names = self._inputs_names

        # Allocate output matrix with number of final ensemble
        # being the multiplication of all ensemble numbers
        nens_final = self._inputs.nens * self._params.nens \
            * self._states.nens * self._nens_outputs

        self._outputs = Matrix.fromdims('outputs',
                nval, noutputs, nens_final)

        self._outputs.names = self._outputs_names[:noutputs]


    def set_uh(self):
        pass


    def initialise(self, states=None, statesuh=None):

        for iens in range(self._nens_states):
            if states is None:
                self._states.iens = iens
                self._states.data = [0.] * self._states.nval

            if statesuh is None:
                self._states.iens = iens
                self._statesuh.data = [0.] * self._statesuh.nval


    def fullrun(self, inputs, params, noutputs=1,
            states=None, statesuh=None,
            idx_start=None, idx_end=None):

        # Set inputs
        inputs_m = Matrix.fromdata('inputs', inputs)
        self.allocate(inputs_m.nval, noutputs)
        self._inputs.data = inputs

        # Set params
        self._params.data = params

        # Initialise model
        self.initialise(states, statesuh)

        if idx_start is None:
            idx_start = 0

        if idx_end is None:
            idx_end = inputs_m.nval - 1

        # Loop through all ensembles
        nens = self.outputs.nens
        for iens in range(nens):

            #iens_inputs =
            #iens_params

            iens_final = iens_inputs * np.prod(nens[:-1]) \
                + iens_params * np.prod(nens[1:-1]) \
                + iens_states * np.prod(nens[2:-1]) \
                + iens_outputs

            self._outputs.iens = iens_final
            self.run1ens(idx_start, idx_end)

        return self.outputs.data[:, :noutputs]


    def run1ens(self, idx_start, idx_end):
        pass


    def clone(self):

        model = Model(
            self._name,
            self._config.nval,
            self._ninputs,
            self._params.nval,
            self._states.nval,
            self._noutputs_max,
            self._inputs_names,
            self._outputs_names,
            self._inputs.nens,
            self._params.nens,
            self._states.nens,
            self._nens_outputs)

        model._params.data = [p.copy() for p in self._params.data]
        model._states.data = [s.copy() for s in self._states.data]
        model._statesuh.data = [u.copy() for u in self._statesuh.data]
        model._config.data = self._config.data.copy()

        if not self._inputs is None:
            model.allocate(self._inputs.nval, self._outputs.nvar)
            model._inputs.data = [i.copy() for i in self._inputs.data]
            model._outputs.data = [o.copy() for o in self._outputs.data]

        return model


