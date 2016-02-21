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
            raise ValueError(('With {0} model, In model {1}, either states, ' +
                'statesuh or inputs' +
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

        # Initialise forecast model
        fmod = self._forecast_model
        fmod.initialise(self.states, self.statesuh)


    def update(self, seed=None):
        ''' Performs states updating '''
        pass

    def check_model(self, periodlength=10):
        ''' Check that simulation model correctly update the outputs
            when states are updated sequentially

        '''

        # get simulation model
        smod = self._sim_model

        # define 2 consecutive periods
        index = smod._inputs.index

        index_start = smod.index_start
        kstart = np.where(index == index_start)[0][0]

        nts = 4
        if smod._inputs.nval < 4:
            raise ValueError(('With model {0}, not enough' +
                ' values in simulation model () to perform' +
                ' check').format(self.name, smod._input.nval))

        index_end = index[kstart + nts]

        kend = np.where(index == index_end)[0][0]
        index_mid = index[(kstart+kend)/2]
        index_midb = index[(kstart+kend)/2+1]

        # Run the model for the first two time steps
        # sequentially
        smod.initialise()
        smod.index_start = index_start
        smod.index_end = index_mid
        smod.run()

        smod.index_start = index_midb
        smod.index_end = index_end
        smod.run()
        o1 = smod.outputs[kend, :].copy()

        # Run the model for the first two time steps
        # jointly
        smod.initialise()
        smod.index_start = index_start
        smod.index_end = index_end
        smod.run()
        o2 = smod.outputs[kend, :].copy()

        # Check that model does not have random outputs
        # by running second simulation twice
        smod.initialise()
        smod.run()
        o3 = smod.outputs[index_end, :].copy()

        # It is expected that both
        # results will be identical
        ck_a = np.allclose(o1, o2)
        ck_b = np.allclose(o2, o3)

        if not ck_a and ck_b:
            raise ValueError(('With model {0}, simulation model does not ' +
                'implement continuous state updating' +
                ' (test for index_start={1} index_end={2})').format(self.name,
                    index_start, index_end))


    def run(self, seed=None):

        # Check model continous state updating
        self.check_model()

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
        index_start = 0
        fc_index = self._inputs.index
        idx_max = np.max(smod._inputs._index)

        for (ifc, index_end) in enumerate(fc_index):

            # Check validity of index
            if index_end-1 > idx_max:
                raise ValueError(('With {0} model, forecast index index_end ({1}) '+
                    'greater than max(input.index)+1 ({2})').format(self.name,
                        index_end, idx_max))

            # Do not run forecast if is outside the forecast model
            # Start/End period
            if not ((index_start >= self.index_start) & (index_end <= self.index_end)):
                continue

            # Set start/end of simulation model
            smod.index_start = index_start
            smod.index_end = index_end-1

            if index_end-1 > index_start:
                raise ValueError(('With {0} model, forecast index index_end ({1}) '+
                    'smaller than index_start+1 ({2})').format(self.name,
                        index_end, index_start+1))

            # Run simulation
            smod.run()

            # Update states and initialise forecast model
            self.update(seed)
            fmod.initialise(smod.states, smod.statesuh)

            # Run forecast for all lead times
            fmod.inputs = self._inputs._data[ifc, :, :, iens_inputs].T
            fmod.run(seed)

            # Store outputs
            self._outputs._data[ifc, :, :, iens_outputs] = fmod.outputs.T

            # Update index for next forecast
            index_start = index_end


    def get_forecast(self, index):
        ''' Extract forecast at index up to lead time = nlead '''

        k = np.where(self._outputs.index == index)[0]
        iens = self.get_iens('outputs')
        fc = self._outputs._data[k, :, :, iens][0].T
        idx = np.arange(index, index+self._outputs.nlead)

        return fc, idx
