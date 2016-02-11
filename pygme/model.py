import math, itertools
import random
import numpy as np

import c_hymod_models_utils

NUHMAXLENGTH = c_hymod_models_utils.uh_getnuhmaxlength()

from pygme.data import Vector, Matrix


class Model(object):

    def __init__(self, name,
            nconfig,
            ninputs,
            nparams,
            nstates,
            noutputs_max,
            nens_params=1,
            nens_states=1,
            nens_outputs=1):

        self.name = name
        self.ninputs = ninputs

        self.nuhlength = 0
        self.noutputs_max = noutputs_max

        # Random ensemble
        self._iens_outputs = 0
        self.nens_outputs = nens_outputs

        self._iens_states = 0
        self.nens_states = nens_states

        # Config vector
        self.config = Vector('config', nconfig)

        # Param vector
        self._params = Vector('params', nparams, nens_params, prefix='X')

        # UH ordinates. Number of ensembles is same than nens_params
        self._uh = Vector('uh', NUHMAXLENGTH, nens_params)
        self._uh.min = np.zeros(NUHMAXLENGTH)
        self._uh.max = np.ones(NUHMAXLENGTH)

        # Initialize UH to [1, 0, 0, .., 0] (neutral UH)
        self._uh.reset(0.)
        for iens in range(nens_params):
            self._uh.iens = iens
            self._uh.data[0] = 1.

        self._nstates = nstates
        self._states = None
        self._statesuh = None

        self._inputs = None
        self._outputs = None

        self._idx_start = 0
        self._idx_end = 0


    def __str__(self):
        str = '\n{0} model implementation\n'.format( \
            self.name)

        for item in ['params', 'states']:
            nval, nens = self.get_dims(item)
            str += '  {0} : {1} val, {2} ens\n'.format(item ,
                                                    nval, nens)

        for item in ['inputs', 'outputs']:
            nval, nvar, nlead, nens = self.get_dims(item)
            str += '  {0} : {1} val, {2} var, {3} lead, {4} ens\n'.format(item ,
                                                    nval, nvar, nlead, nens)
        str += '  nuhlengthmax    = {0}\n'.format(self._uh.nval)

        return str


    @property
    def ilead(self):
        if self._inputs is None:
            raise ValueError(('Model {0}: Cannot get ilead when'+
                ' inputs is None. Please allocate').format(self.name))

        return self._inputs.ilead

    @ilead.setter
    def ilead(self, value):
        if self._inputs is None:
            raise ValueError(('Model {0}: Cannot set ilead when'+
                ' inputs is None. Please allocate').format(self.name))

        self._inputs.ilead = value
        self._outputs.ilead = value


    @property
    def idx_start(self):
        return self._idx_start

    @idx_start.setter
    def idx_start(self, value):
        if self._inputs is None:
            raise ValueError(('Model {0}: Cannot set idx_end when'+
                ' inputs is None. Please allocate').format(self.name))

        if value < 0 or value >= self._inputs.nval:
            raise ValueError(('Model {0}: idx_start < 0 or > inputs.nval ({1})').format(self.name,
                self._inputs.nval))

        self._idx_start = value


    @property
    def idx_end(self):
        return self._idx_end

    @idx_end.setter
    def idx_end(self, value):
        if self._inputs is None:
            raise ValueError(('Model {0}: Cannot set idx_end when'+
                ' inputs is None. Please allocate').format(self.name))

        if value < 0 or value >= self._inputs.nval:
            raise ValueError(('Model {0}: idx_end < 0 or > inputs.nval ({1})').format(self.name,
                self._inputs.nval))

        self._idx_end = value


    @property
    def inputs(self):
        return self._inputs.data

    @inputs.setter
    def inputs(self, value):
        self._inputs.data = value


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
    def uh(self):
        return self._uh.data

    @uh.setter
    def uh(self, value):
        ''' Set UH values '''
        if np.abs(np.sum(value)-1.) > 1e-9:
            raise ValueError(('Model {0}: Trying to set uhdata '
                    'that do not sum to 1 ({1})').format( \
                                self.name, np.sum(value)))
        self._uh.data = value


    @property
    def states(self):
        if self._states is None:
            raise ValueError(('Model {0}: Cannot set states when'+
                ' states object is None. Please allocate').format(self.name))

        # Code needed to sync input/params ensemble with states
        self._statesuh.iens = self._states.iens
        return self._states.data

    @states.setter
    def states(self, value):
        self._states.data = value


    @property
    def statesuh(self):
        if self._statesuh is None:
            raise ValueError(('Model {0}: Cannot set statesuh when'+
                ' statesuh object is None. Please allocate').format(self.name))

        # Code needed to sync input/params ensemble with states
        self._statesuh.iens = self.iens_states
        return self._statesuh.data

    @statesuh.setter
    def statesuh(self, value):
        self._statesuh.data = value


    @property
    def outputs(self):
        return self._outputs.data

    @outputs.setter
    def outputs(self, value):
        self._outputs.data = value


    @property
    def iens_outputs(self):
        return self._iens_outputs


    def get_iens(item='params'):
        ''' Set ensemble member for model attributes '''
        if item in ['params', 'inputs']:
            obj = getattr(self, '_{0}'.format(item))
            if not obj is None:
                return obj.iens
            else:
                raise ValueError(('Model {0}: getting ensemble number, but item {1}' +
                ' is None. Cannot get ensemble number').format(self.name, item))

        elif item == 'states':
            return self._iens_states

        elif item == 'outputs':
           return self._outputs_states

        else:
            raise ValueError(('Model {0}: setting ensemble number, but item {1}' +
                ' does not exists').format(self.name, item))



    def set_iens(self, item='params', value=0):
        ''' Set ensemble member for model attributes '''
        value = int(value)

        if item in ['params', 'inputs']:
            obj = getattr(self, '_{0}'.format(item))
            if not obj is None:
                obj.iens = value
            else:
                raise ValueError(('Model {0}: setting ensemble number, but item {1}' +
                ' is None. Cannot get ensemble number').format(self.name, item))


        elif item == 'states':
            if value >= self.nens_states or value < 0:
                raise ValueError(('Model {0}: iens_states {1} ' \
                        '>= nens {2} or < 0').format( \
                                    self.name, value, self.nens_states))

            # Determines the state ensemble number given the input ensemble and the parameter ensemble
            # It is assumed that the loop will go first on inputs then on parameters
            # and finally on random states ensembles
            n1 = self._params.nens
            n2 = self.nens_states
            iens = n1 * n2  * self._inputs.iens
            iens += n2 * self._params.iens
            iens += value

            # Set ensemble number
            self._iens_states = value
            self._states.iens = iens
            self._statesuh.iens = iens

        elif item == 'outputs':
            if value >= self.nens_outputs or value < 0:
                raise ValueError(('Model {0}: iens_outputs {1} ' \
                        '>= nens {2} or < 0').format( \
                                    self.name, value, self.nens_outputs))

            # Determines the state ensemble number given the input ensemble and the parameter ensemble
            # It is assumed that the loop will go first on inputs then on parameters
            # and finally on random states ensembles
            n1 = self._params.nens
            n2 = self.nens_states
            n3 = self.nens_outputs
            iens = n1 * n2  * n3 * self._inputs.iens
            iens += n2 * n3 * self._params.iens
            iens += n3 * self._iens_states
            iens += value

            # Set ensemble number
            self._outputs_states = value

        else:
            raise ValueError(('Model {0}: setting ensemble number, but item {1}' +
                ' does not exists').format(self.name, item))


    def get_dims(self, item='params'):
        ''' Getting dimensions of object '''

        nval = None
        nvar = None
        nlead = None
        nens = None

        if item == 'inputs':
            if not self._inputs is None:
                nval = self._inputs.nval
                nvar = self._inputs.nvar
                nlead = self._inputs.nlead
                nens = self._inputs.nens

            return (nval, nvar, nlead, nens)

        elif item == 'params':
            nens = self._params.nens
            nval = self._params.nval

            return (nval, nens)

        elif item == 'states':
            if not self._states is None:
                nens = self.nens_states
                nval = self._states.nval

            return (nval, nens)

        elif item == 'outputs':
            if not self._outputs is None:
                nval = self._outputs.nval
                nvar = self._outputs.nvar
                nlead = self._outputs.nlead
                nens = self.nens_outputs

            return (nval, nvar, nlead, nens)

        else:
            raise ValueError(('Model {0}: getting dims, but item {1}' +
                ' does not exists').format(self.name, item))


    def allocate(self, nval, noutputs=1, nlead_inputs=1, nens_inputs=1):
        ''' We define the number of outputs here to allow more flexible memory allocation '''

        if noutputs <= 0:
            raise ValueError(('Number of outputs defined' + \
                ' for model {0} should be >0').format(nval))

        if noutputs > self.noutputs_max:
            raise ValueError(('Too many outputs defined for model {0}:' + \
                ' noutputs({1}) > noutputs_max({2})').format( \
                self.name, noutputs, self.noutputs_max))

        self._inputs = Matrix.from_dims('inputs',
                nval, self.ninputs, nlead_inputs, nens_inputs, prefix='I')

        # Allocate state vectors with number of ensemble
        nens = self._params.nens * self._inputs.nens * self.nens_states
        self._states = Vector('states', self._nstates, nens,
                                prefix='S', has_minmax=False)

        self._statesuh = Vector('statesuh', NUHMAXLENGTH, nens,
                                prefix='SUH')
        self._statesuh.min = np.zeros(NUHMAXLENGTH)

        # Allocate output matrix with number of final ensemble
        nens *= self.nens_outputs
        self._outputs = Matrix.from_dims('outputs',
                nval, noutputs, nlead_inputs, nens, prefix='O')

        # Set up start and end to beginning and end of simulation
        self.idx_start = 0
        self.idx_end = nval-1


    def set_uh(self):
        pass


    def initialise(self, states=None, statesuh=None):

        if self._states is None:
            raise ValueError(('Model {0}: Cannot initialise when'+
                ' states is None. Please allocate').format(self.name))

        for iens in range(self._states.nens):
            self._states.iens = iens
            self._statesuh.iens = iens

            if states is None:
                self._states.data = [0.] * self._states.nval
            else:
                self._states.data = states

            if statesuh is None:
                self._statesuh.data = [0.] * self._statesuh.nval
            else:
                self._statesuh.data = statesuh


    def reset(self, item='params', value=None):
        ''' Function to reset model objects '''
        obj = getattr(self, '_{0}'.format(item))

        if obj is None or obj is None:
            raise ValueError(('Model {0}: Model does not have attribute _{1}').format( \
                                self.name, item))

        setattr(self, item, obj.default)


    def random(self, item='params', distribution='normal', seed=3):
        ''' Function to randomise model items '''
        obj = getattr(self, '_{0}'.format(item))

        if obj is None:
            raise ValueError(('Model {0}: Model does not have object {1}').format( \
                                self.name, item))

        obj.random(distribution, seed)


    def run(self):
        pass

    def run_ens(self,
            ens_inputs = None,
            ens_params = None,
            ens_states = None,
            ens_outputs = None):
        ''' Run the model with selected ensembles '''

        # Set ensemble lists
        if ens_inputs is None:
            ens_inputs = range(self._inputs.nens)

        if ens_params is None:
            ens_params = range(self._params.nens)

        if ens_states is None:
            ens_states = range(self.nens_states)

        if ens_outputs is None:
            ens_outputs = range(self.nens_outputs)

        # Loop through ensembles
        for i_inputs, i_params, i_states, i_outputs in \
                itertools.product(ens_inputs, ens_params,
                        ens_states, ens_outputs):

            self.set_iens('inputs', i_inputs)
            self.set_iens('params', i_params)
            self.set_iens('states', i_states)
            self.set_iens('outputs', i_outputs)

            self.run()


    def clone(self):
        ''' Clone the current object model '''

        model = Model(self.name,
            self.config.nval,
            self.ninputs,
            self._params.nval,
            self._nstates,
            self.noutputs_max,
            self._params.nens,
            self.nens_states,
            self.nens_outputs)

        for item in ['_params', '_uh', '_states',
            '_statesuh', 'config', '_inputs', '_outputs']:
            obj = getattr(self, item)

            if not obj is None:
                setattr(model, item, obj.clone())

        return model


