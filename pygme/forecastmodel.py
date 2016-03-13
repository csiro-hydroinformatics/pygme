import math
import numpy as np

from pygme.model import Model
from pygme.data import Matrix

def get_perfect_forecast(matrix, nlead=2):
    ''' Turn a standard input matrix into a perfect forecast matrix '''

    if not matrix.index_continuous:
        raise ValueError('Perfect forecast cannot be produced' +
                ' for non continuous matrix index')

    nval = matrix.nval
    nvar = matrix.nvar
    nens = matrix.nens
    fc = Matrix.from_dims('fc', nval, nvar, nlead, nens,
            index=matrix.index)

    for l in range(nens):
        matrix.iens = l

        for k in range(nlead):
            data = np.roll(matrix.data, -k, 0)
            data[range(nval-k, nval), :] = np.nan
            fc.ilead = k
            fc.iens = l
            fc.data = data

    fc.ilead = 0

    return fc



class ForecastModel(Model):

    def __init__(self, model,
            ninputs=None, nparams=None, nstates=None,
            nens_params=1,
            nens_states=1,
            nens_outputs=1):

        name = '{0}-forecast'.format(model.name)

        # Check sim model is allocated
        if model._states is None or model._statesuh is None \
            or model._inputs is None:
            raise ValueError(('With {0} model, In model {1}, either states, ' +
                'statesuh or inputs' +
                '  is None. Please allocate').format(name, model.name))

        self._sim_model = model
        self._forecast_model = model.clone()

        # Affect model dimensions to forecast model if not defined
        if ninputs is None:
            _, ninputs, _, _ = model.get_dims('inputs')

        if nparams is None:
            nparams, _ = model.get_dims('params')

        if nstates is None:
            nstates, _ = model.get_dims('states')


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


    def allocate(self, inputs, noutputs=None):

        if noutputs is None:
            _, noutputs, _, _ = self._sim_model.get_dims('outputs')

        # Allocate self
        super(ForecastModel, self).allocate(inputs, noutputs)

        # Allocate forecast model
        # i.e. model with nval= nlead-1 to run model over
        # the whole forecast period excluding the current instant
        _, nvar, nlead, nens = self.get_dims('inputs')

        if nlead <= 1:
            raise ValueError('With {0} model, cannot allocate a ' +
                    'forecast model with inputs having nlead({1}) <= 1'.format(
                        self.name, nlead))

        inputs = Matrix.from_dims('fc_inputs', nlead-1, nvar, 1, nens,
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

        if not states is None:
            states = self.states[:nstates]
        if not statesuh is None:
            staesuh = self.statesuh[:nstatesuh]

        smod.initialise(states, statesuh)


    def update(self, seed=None):
        ''' Performs states updating '''
        pass


    def check_model(self, periodlength=10):
        ''' Check that simulation model correctly update the outputs
            when states are updated sequentially

        '''
        # get simulation model
        smod = self._sim_model
        states = smod.states.copy()
        statesuh = smod.statesuh.copy()

        if smod._inputs.nval < periodlength:
            raise ValueError(('With model {0}, not enough' +
                ' values in simulation model () to perform' +
                ' check').format(self.name, smod._input.nval))

        # define 2 consecutive periods
        index = smod._inputs.index

        index_start = smod.index_start
        kstart = np.where(index == index_start)[0][0]

        index_end = index[kstart + periodlength]
        index_mid = index[kstart + periodlength/2]
        index_midb = index[kstart + periodlength/2 + 1]

        kend = np.where(index == index_end)[0][0]

        # Run the model in two steps
        smod.initialise(states=states, statesuh=statesuh)
        smod.index_start = index_start
        smod.index_end = index_mid
        smod.run()

        smod.index_start = index_midb
        smod.index_end = index_end
        smod.run()
        o1 = smod.outputs[kend, :].copy()

        # Run the model over the whole period
        smod.initialise(states=states, statesuh=statesuh)
        smod.index_start = index_start
        smod.index_end = index_end
        smod.run()
        o2 = smod.outputs[kend, :].copy()

        # Check that model does not have random outputs
        # by running the whole period twice
        smod.initialise(states=states, statesuh=statesuh)
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

        # Reset model to previous state
        smod.initialise(states=states, statesuh=statesuh)


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

        # Loop through forecast time indexes
        index_start = 0
        fc_index = self._inputs.index
        sim_index = self._sim_model._inputs.index
        idx_max = np.max(smod._inputs._index)

        for (ifc, index_end) in enumerate(fc_index):
            # Check validity of index
            if index_end > idx_max:
                raise ValueError(('With {0} model, forecast index index_end ({1}) '+
                    'greater than max(input.index)+1 ({2})').format(self.name,
                        index_end, idx_max))

            # Do not run forecast if is outside the forecast model
            # Start/End period
            if not ((index_end >= self.index_start) & (index_end <= self.index_end)):
                continue

            # Set start/end of simulation model
            smod.index_start = index_start
            smod.index_end = index_end

            # Run simulation
            smod.run()

            # Store outputs in simulation mode
            # i.e. ilead = 0
            kend = np.where(index_end == sim_index)[0][0]
            self._outputs._data[ifc, :, 0, iens_outputs] = smod.outputs[kend, :]

            # Update states and initialise forecast model
            self.update(seed)
            fmod.initialise(smod.states, smod.statesuh)

            # Run forecast for all lead times
            finputs = np.transpose(self._inputs._data[ifc, :, 1:, iens_inputs])
            fmod.inputs = finputs
            fmod.run(seed)

            # Store outputs in forecast mode
            # Forecast data starts when ilead >= 1 !!!
            # ilead = 0 is the simulation mode
            self._outputs._data[ifc, :, 1:, iens_outputs] = fmod.outputs.T

            # Update index for next forecast
            index_start = index_end+1


    def get_forecast(self, index):
        ''' Extract forecast at index up to lead time = nlead. First value is the simulation mode '''

        # Forecast index
        # Use the index defined for the simulation model
        sim_index = self._sim_model._inputs.index
        istart = np.where(sim_index == index)[0][0]

        nval = self._sim_model._inputs.nval
        iend = min(nval, istart+self._outputs.nlead)

        forc_data_index = sim_index[range(istart, iend)]

        # Fill up with nan if we reach the end of the index vector
        forc_data_index = np.concatenate([forc_data_index,
            [np.nan] * (self._outputs.nlead - len(forc_data_index))]).astype(np.int32)

        # Extract forecast data
        forc_index = self._outputs.index
        iforc = np.where(forc_index == index)[0][0]
        data = self._outputs._data[iforc, :, :, :].transpose((1, 0, 2))

        forc_data = Matrix.from_data('{0}_forecasts'.format(self.sim_model.name),
                data, index=forc_data_index)

        return forc_data

