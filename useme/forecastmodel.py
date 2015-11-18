import math
import numpy as np

from useme.model import Model, Matrix


class ForecastData(object):

    def __init__(self, nlead):
        self._nlead = nlead
        self._inputs = None
        self._outputs = None
        self._states = None
        self._statesuh = None

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, value):
        if self._inputs is None:
            self._inputs = Matrix.fromdata('inputs', value)

            nval = self._inputs.nval
            if nval != self._nlead:
                raise ValueError(('Number of values in input data({0}) ' + \
                    'does not match nlead({1})').format(nval, self._nlead))
        else:
            self._inputs.data = value


    @property
    def outputs(self):
        return self._outputs

    @outputs.setter
    def outputs(self, value):
        if self._outputs is None:
            self._outputs = Matrix.fromdata('outputs', value)

            nval = self._outputs.nval
            if nval != self._nlead:
                raise ValueError(('Number of values in output data({0}) ' + \
                    'does not match nlead({1})').format(nval, self._nlead))
        else:
            self._outputs.data = value


    @property
    def states(self):
        return self._states

    @states.setter
    def states(self, value):
        if self._states is None:
            self._states = Vector('states', len(value))

        self._states.data = value


    @statesuh.setter
    def statesuh(self, value):
        if self._statesuh is None:
            self._statesuh = Vector('statesuh', len(value))

        self._statesuh.data = value



class ForecastModel(Model):

    def __init__(self, model, nwarmup, nlead, nparams)

        self._model = model
        self._nwarmup = nwarmup
        self._nlead = nlead

        outputs_names = ['F{0:03d}'.format(i) for i in range(nlead)]

        Model.__init__(self,
            name='{0}-forecast'.format(model.name),
            nconfig=1, \
            ninputs=1, \
            nparams=nparams, \
            nstates=1, \
            noutputs_max = nlead, \
            inputs_names = [''], \
            outputs_names = outputs_names)

        self._fcdata = {}


    def __str__(self):

        str = '\n{0} Forecast model implementation\n'.format( \
            self._name)
        str += '  ninputs      = {0}\n'.format(self._model._ninputs)
        str += '  nuhmaxlength = {0}\n'.format(self._model._statesuh.nval)
        str += '  nuhlength    = {0}\n'.format(self._model._nuhlength)

        str += '  {0}\n'.format(self._model._config)
        str += '  {0}\n'.format(self._model._states)
        str += '  {0}\n'.format(self._model._params)

        if not self._inputs is None:
            str += '  {0}\n'.format(self._model._inputs)

        if not self._outputs is None:
            str += '  {0}\n'.format(self._model._outputs)

        return str


    @property
    def nlead(self):
        return self._nlead


    @property
    def model(self):
        return self._model


    @property
    def fcdata(self):
        return self._fcdata


    def allocate_forecast_data(self, idx, iens):
        self._fcdata[(idx, iens)] = ForecastData(self._nlead)


    def run(self):

        # Run model
        self._model.run()


        # Loop through forecast data
        for k in self._fcdata:
            states =
        pass


    def clone(self):
        model_clone = self._model.clone()




