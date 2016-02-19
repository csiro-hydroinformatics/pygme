import os
import re
import unittest

import time

import numpy as np

from pygme import calibration
from pygme.forecastmodel import ForecastModel
from pygme.models.gr4j import GR4J
from pygme.data import Matrix

from dummy import Dummy, MassiveDummy

class ForecastModelTestCases(unittest.TestCase):


    def setUp(self):
        print('\t=> ForecastModelTestCase')
        filename = os.path.abspath(__file__)
        self.FHERE = os.path.dirname(filename)


    def test_initialise(self):
        gr = GR4J()
        gr.allocate(np.random.uniform(0, 1, (10, 2)), 1)
        fc = ForecastModel(gr, 2, 4, 2)
        str_fc = '%s' % fc

        try:
            fc = ForecastModel(gr, 1, 4, 2)
        except ValueError, e:
            pass
        self.assertTrue(str(e).startswith('With gr4j#forecast model, ' +
            'Number of inputs'))

        try:
            fc = ForecastModel(gr, 2, 3, 1)
        except ValueError, e:
            pass
        self.assertTrue(str(e).startswith('With gr4j#forecast model, ' +
            'Number of parameters'))

        try:
            fc = ForecastModel(gr, 2, 4, 1)
        except ValueError, e:
            pass
        self.assertTrue(str(e).startswith('With gr4j#forecast model, ' +
            'Number of states'))

    def test_continuous(self):
        # Prepare simulation model
        nval = 1000
        sim_inputs = np.random.uniform(0, 1, (nval, 2))
        dum = Dummy()
        dum.allocate(sim_inputs, 2)

        # Prepare forecast inputs
        fc_nval = 10
        nlead = 5
        index = np.arange(1, fc_nval+1)
        fc_inputs = Matrix.from_dims('fc', fc_nval,
                2, nlead, index=index)

        # Run forecasts
        dum.config['continuous'] = 0
        fc = ForecastModel(dum, 2, 3, 2)
        fc.allocate(fc_inputs, 2)
        fc.params = [1, 10, 0.]
        fc.initialise()

        try:
            fc.run()
        except ValueError, e:
            pass

        self.assertTrue(str(e).startswith('With model dummy#forecast, ' +
                'simulation model does not implement'))


    def test_run1(self):
        # Prepare simulation model
        dum = Dummy()
        dum.config['continuous'] = 1
        nval = 1000
        sim_inputs = np.random.uniform(0, 1, (nval, 2))
        dum.allocate(sim_inputs, 2)

        params = [0.5, 10., 0.]
        dum.params = params

        states = [10, 0]
        dum.initialise(states=states)
        dum.run()
        expected = dum.outputs.copy()
        #dum._outputs.reset(np.nan)

        # Prepare forecast inputs
        fc_nval = 10
        nlead = 5
        #index = np.arange(0, nval, nval/fc_nval)
        index = np.arange(1, fc_nval+1)
        fc_inputs = Matrix.from_dims('fc', fc_nval,
                2, nlead, index=index)

        for k in range(nlead):
            fc_inputs.ilead = k
            fc_inputs.data = sim_inputs[index+k+1, :]

        # Run forecasts
        fc = ForecastModel(dum, 2, 3, 2)
        fc.allocate(fc_inputs, 2)
        fc.params = params
        fc.initialise(states=states)
        fc.run()

        # Check forecasts
        for k in range(fc_nval):
            res, idx = fc.get_forecast(index[k])
            err = np.abs(expected[idx, :] - res)



    def test_run2(self):
        return
        warmup = 365 * 5
        gr = GR4J()

        fp = '{0}/data/GR4J_params.csv'.format(self.FHERE)
        params = np.loadtxt(fp, delimiter=',')

        for i in range(params.shape[0]):

            fts = '{0}/data/GR4J_timeseries_{1:02d}.csv'.format( \
                    self.FHERE, i+1)
            data = np.loadtxt(fts, delimiter=',')
            inputs = np.ascontiguousarray(data[:, [1, 2]], np.float64)

            # Run gr4j
            gr.allocate(inputs, 1)

            gr.params = params[i, [2, 0, 1, 3]]
            gr.initialise()
            gr.run()
            qsim = gr.outputs[:,0]



