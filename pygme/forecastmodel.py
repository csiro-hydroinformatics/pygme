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
            raise ValueError(('Model {0}: In model {1}, either states, statesuh or inputs' +
                '  is None. Please allocate').format(name, model.name))

        self._sim_model = model
        self._forecast_model = model.clone()

        # Check dimensions
        _, ninputs2, _, _ = model.get_dims('inputs')
        if ninputs < ninputs2:
            raise ValueError(('Model {0}: Number of inputs in forecast ' +
                'model ({1}) lower than number of model'+
                ' inputs ({2})').format(name, ninputs, ninputs2))

        nparams_model, _ = model.get_dims('params')
        if nparams < nparams_model:
            raise ValueError(('Model {0}: Number of parameters in forecast ' +
                'model ({1}) lower than number of model'+
                ' parameters ({2})').format(name, nparams, nparams_model))

        _, nstates2 = model.get_dims('states')
        if nstates < model._nstates:
            raise ValueError(('Model {0}: Number of states in forecast ' +
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


    def allocate(self, nval, noutputs=1, nlead_inputs=1, nens_inputs=1):

        # Allocate self
        super(ForecastModel, self).allocate(nval,
                noutputs, nlead_inputs, nens_inputs)

        # Allocate forecast model
        # i.e. model with nval= nlead to run model over
        # the whole forecast period
        self._forecast_model.allocate(nlead_inputs, noutputs,
                1, nens_inputs)

        # Sim model is expected to be allocated outside !


    def update(self, seed):
        ''' Performs states updating '''
        pass



    def run(self, seed):

        # Get models
        smod = self._sim_model
        fmod = self._forecast_model

        # ensemble numbers
        iens_inputs = self._inputs.iens
        iens_outputs = self._outputs.iens

        # Check model inputs are continuous ts_index
        if not smod._inputs.ts_index_continuous:
            raise ValueError(('Model {0}: Simulation model should have' +
                'inputs with continuous ts_index'.format(self.name)))

        if not smod._inputs.ts_index[0] != 0:
            raise ValueError(('Model {0}: Simulation model should have' +
                'inputs with continuous ts_index starting at idx=0' +
                ' (currently {1})').format(self.name,
                    smod._inputs.ts_index[0]))

        # Loop through forecast time indexes
        idx_start = 0
        idx_max = np.max(sim_ts_index)

        for (ifc, idx_end) in enumerate(fmod._inputs.ts_index):

            # Check validity of ts_index
            if idx_end > idx_max:
                raise ValueError(('Model {0}: Tried forecasting for ts_index {1}' +
                    ', but simulation model has a max ts_index' +
                    ' equal to {2}').format(self.name, idx_end, idx_max))

            if not ((idx_start >= self.idx_start) & (idx_end <= self.idx_end)):
                continue

            # Set start/end of simulation
            self._sim_model.idx_start = idx_start
            self._sim_model.idx_end = idx_end

            # Run simulation
            self._sim_model.run()
            self.sim_states[ifc, :] = self._sim_model.states
            self.sim_statesuh[ifc, :] = self._sim_model.statesuh

            # Update states and initialise forecast model
            self.update(seed)
            self._forecast_model.initialise(smod.states, smod.statesuh)

            # Run forecast for all lead times
            fmod.inputs = self._inputs.data[ifc, :, :, iens_inputs].T
            fmod.run(seed)

            # Store outputs
            fmod._outputs.data[ifc, :, :, iens_outputs] = fmod.outputs.T


    def clone(self):
        pass




