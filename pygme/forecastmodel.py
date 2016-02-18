import math
import numpy as np

from pygme.model import Model
from pygme.data import Matrix


class ForecastModel(Model):

    def __init__(self, model,
            ninputs, nparams, nstates,
            nens_params=1,
            nens_states=1,
            nens_outputs=1):

        name = '{0}#forecast'.format(model.name)

        # Check sim model is allocated
        if model._states is None or model._statesuh is None \
            or model._inputs is None:
            raise ValueError(('With {0} model, In model {1}, either states, statesuh or inputs' +
                '  is None. Please allocate').format(name, model.name))

        self._sim_model = model
        self._forecast_model = model.clone()

        # Check dimensions
        _, ninputs2, _, _ = model.get_dims('inputs')
        if ninputs < ninputs2:
            raise ValueError(('With {0} model, Number of inputs in forecast ' +
                'model ({1}) lower than number of model'+
                ' inputs ({2})').format(name, ninputs, ninputs2))

        nparams_model, _ = model.get_dims('params')
        if nparams < nparams_model:
            raise ValueError(('With {0} model, Number of parameters in forecast ' +
                'model ({1}) lower than number of model'+
                ' parameters ({2})').format(name, nparams, nparams_model))

        _, nstates2 = model.get_dims('states')
        if nstates < model._nstates:
            raise ValueError(('With {0} model, Number of states in forecast ' +
                'model ({1}) lower than number of model'+
                ' parameters ({2})').format(name, nstates, nstates2))

        # Initialise model
        nconfig, _ = model.get_dims('config')

        Model.__init__(self,
            name=name,
            nconfig=nconfig,
            ninputs=ninputs,
            nparams=nparams,
            nstates=nstates,
            noutputs_max = model.noutputs_max,
            nens_params=nens_params,
            nens_states=nens_states,
            nens_outputs=nens_outputs)


    @property
    def sim_model(self):
        return self._sim_model

    @property
    def forecast_model(self):
        return self._forecast_model

    @property
    def sim_states(self):
        return self._sim_states.data

    @property
    def sim_statesuh(self):
        return self._sim_statesuh.data

    @property
    def params(self):
        return self.params

    @params.setter
    def params(self, value):
        nparams_model, _ = self._sim_model.get_dims('params')

        self._sim_model.params = value[:nparams_model]
        self._forecast_model.params = value[:nparams_model]


    def allocate(self, inputs, noutputs=1):

        # Allocate self
        super(ForecastModel, self).allocate(inputs, noutputs)

        # Allocate forecast model
        # i.e. model with nval= nlead to run model over
        # the whole forecast period
        _, nvar, nlead, nens = self.get_dims('inputs')
        inputs = Matrix.from_dims('fc_inputs', nlead, nvar, 1, nens,
                prefix='FCI')
        self._forecast_model.allocate(inputs, noutputs)

        # Sim model is expected to be allocated outside !
        # Nothing to do here


    def initialise(self, states=None, statesuh=None):

        super(ForecastModel, self).initialise(states, statesuh)

        # Initialise simulation model
        smod = self._sim_model
        nstates, _ = smod.get_dims('states')
        nstatesuh, _ = smod.get_dims('statesuh')

        smod.initialise(self.states[:nstates],
                self.statesuh[:nstatesuh])


    def update(self, seed):
        ''' Performs states updating '''
        pass


    def run(self, seed=None):

        # Get models
        smod = self._sim_model
        fmod = self._forecast_model

        # ensemble numbers
        iens_inputs = self._inputs.iens
        iens_outputs = self._outputs.iens

        # Check model inputs are continuous index
        if not smod._inputs.index_continuous:
            raise ValueError(('With {0} model, Simulation model should have' +
                'inputs with continuous index'.format(self.name)))

        if smod._inputs.index[0] != 0:
            raise ValueError(('With {0} model, Simulation model should have' +
                ' inputs with continuous index starting at idx=0' +
                ' (currently {1})').format(self.name,
                    smod._inputs.index[0]))

        # Loop through forecast time indexes
        idx_start = 0
        fc_index = self._inputs.index
        idx_max = np.max(smod._inputs._index)

        for (ifc, idx_end) in enumerate(fc_index):

            # Check validity of index
            if idx_end > idx_max:
                raise ValueError(('With {0} model, forecast index idx_end ({1}) '+
                    'greater than max(input.index) ({2})').format(self.name,
                        idx_end, idx_max))

            if not ((idx_start >= self.idx_start) & (idx_end <= self.idx_end)):
                continue

            if idx_end-1 >= idx_start:
                # Set start/end of simulation
                smod.idx_start = idx_start
                smod.idx_end = idx_end-1

                # Run simulation
                smod.run()

            # Update states and initialise forecast model
            self.update(seed)
            fmod.initialise(smod.states, smod.statesuh)
            print(ifc, idx_start, idx_end, smod.states[0])

            # Run forecast for all lead times
            fmod.inputs = self._inputs._data[ifc, :, :, iens_inputs].T
            fmod.run(seed=-2)

            # Store outputs
            self._outputs._data[ifc, :, :, iens_outputs] = fmod.outputs.T

            # Update index for next forecast
            idx_start = idx_end



