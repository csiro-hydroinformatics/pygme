import os
import re
import unittest

import time

import numpy as np

from pygme import calibration
from pygme.forecastmodel import ForecastModel, get_perfect_forecast
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
        self.assertTrue(str(e).startswith('With gr4j-forecast model, ' +
            'Number of inputs'))

        try:
            fc = ForecastModel(gr, 2, 3, 1)
        except ValueError, e:
            pass
        self.assertTrue(str(e).startswith('With gr4j-forecast model, ' +
            'Number of parameters'))

        try:
            fc = ForecastModel(gr, 2, 4, 1)
        except ValueError, e:
            pass
        self.assertTrue(str(e).startswith('With gr4j-forecast model, ' +
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

        self.assertTrue(str(e).startswith(('With model {0}, ' +
                'simulation model does not implement').format(fc.name)))

    def test_perfect_forecast(self):
        nval = 100
        nvar = 3
        inputs = Matrix.from_data('inputs', np.random.uniform(size=(nval, nvar)))

        nlead = 20
        fc_inputs = get_perfect_forecast(inputs, nlead)

        for k in range(nlead):
            v = inputs.data[range(k, nval), :]
            v = np.concatenate([v, np.nan * np.ones((k, nvar))], 0)

            fc_inputs.ilead = k
            kk = range(nval-k, k)
            ck = np.allclose(fc_inputs.data[kk], v[kk])
            self.assertTrue(ck)


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

        # Prepare forecast inputs
        fc_nval = 10
        nlead = 5
        index = np.arange(0, nval, nval/fc_nval)

        fc_inputs = get_perfect_forecast(dum._inputs, nlead)
        fc_inputs = fc_inputs.slice(index)

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

            ck = np.allclose(err, 0.)
            self.assertTrue(ck)


    def test_run2(self):
        warmup = 365 * 5
        gr = GR4J()

        fp = '{0}/data/GR4J_params.csv'.format(self.FHERE)
        params = np.loadtxt(fp, delimiter=',')

        for i in range(params.shape[0]):

            fts = '{0}/data/GR4J_timeseries_{1:02d}.csv'.format( \
                    self.FHERE, i+1)
            data = np.loadtxt(fts, delimiter=',')
            inputs = np.ascontiguousarray(data[:, [1, 2]], np.float64)
            nval = inputs.shape[0]

            # Run gr4j
            nout = 1
            gr.allocate(inputs, nout)

            pp = params[i, [2, 0, 1, 3]]
            gr.params = pp
            gr.initialise()
            gr.run()
            qsim = gr.outputs[:,0]

            # Create input forecast matrix
            # -> 30 day forecast, produced every week
            nlead = 30
            nfc = nval/7

            fc_inputs = get_perfect_forecast(gr._inputs, nlead)
            index = np.arange(0, nval, nval/nfc)
            index = index[index+nlead <= nval]
            nfc = len(index)

            fc_inputs = fc_inputs.slice(index)

            # Create forecast model
            fc = ForecastModel(gr)
            fc.allocate(fc_inputs, nout)

            # Run
            t0 = time.time()

            fc.params = pp
            fc.initialise()
            fc.run()

            t1 = time.time()
            dta = 1000 * (t1-t0)
            dta2 = dta/nlead/nfc * 365.25
            print(('\t\tTEST {0:2d} : nlead={2}, nfc={3} ~ runtime' +
                ' {1:0.5f}ms/yr (total {4:0.5}ms)').format(i+1,
                    dta2, nlead, nfc, dta))

            # Check output
            for i in index:
                res, idx = fc.get_forecast(i)
                expected = qsim[idx]
                ck = np.allclose(res.flat[:], expected)




