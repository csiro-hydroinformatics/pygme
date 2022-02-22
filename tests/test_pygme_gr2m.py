import os, sys
import re
import unittest

from timeit import Timer
import time

import numpy as np
from pygme.models.gr2m import GR2M, CalibrationGR2M
from pygme.calibration import ObjFunSSE

import warnings

import testdata

class GR2MTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> GR2MTestCase')
        filename = os.path.abspath(__file__)
        self.ftest = os.path.dirname(filename)


    def test_print(self):
        gr = GR2M()
        str_gr = '%s' % gr


    def test_gr2m_dumb(self):
        gr = GR2M()

        nval = 100
        p = np.exp(np.random.normal(0, 2, size=nval))
        pe = np.ones(nval) * 5.
        inputs = np.concatenate([p[:,None], pe[:, None]], axis=1)
        gr.allocate(inputs, 12)

        gr.X1 = 400
        gr.X2 = 0.9
        gr.initialise()

        gr.run()
        df = gr.outputs_dataframe()


    def test_run_against_data(self):
        ''' Compare GR2M simulation with test data '''
        warmup = 12 * 12
        gr = GR2M()

        for i in range(20):
            if i in [2, 9, 14]:
                warnings.warn('Skipping test {0} for GR2m'.format(i+1))
                continue

            data = testdata.read('GR2M_timeseries_{0:02d}.csv'.format(i+1), \
                                    source='output', has_dates=False)
            params = testdata.read('GR2M_params_{0:02d}.csv'.format(i+1), \
                                    source='output', has_dates=False)

            inputs = np.ascontiguousarray(\
                            data.loc[:, ['Precip', 'PotEvap']], \
                            np.float64)

            # Run gr4j
            gr.allocate(inputs, 10)
            gr.params.values = params

            # .. initiase to same values than IRSTEA run ...
            # Estimate initial states based on first two state values
            #s0 = data[0, [6, 7]]
            #s1 = data[1, [6, 7]]
            #sini = 2*s0-s1

            # Other initial condition from GR2m Excel
            sini = [gr.params.X1/2, 30]
            gr.initialise(states=sini)

            gr.run()

            # Compare
            idx = np.arange(inputs.shape[0]) > warmup
            #sim = gr.outputs[:, [0, 3, 6, 9]].copy()
            #expected = data[:, [8, 5, 4, 2]]
            sim = gr.outputs[:, 0][:, None].copy()
            expected = data.loc[:, 'Qsim2'].values[:, None]
            err = np.abs(sim[idx, :] - expected[idx, :])

            # Sensitivity to initial conditionos
            s1 = [0]*2
            s2 = [gr.params.X1, 60]
            warmup_ideal, sim0, sim1 = gr.inisens(s1, s2)

            # Special criteria
            # 5 values with difference greater than 1e-5
            # max diff lower than 5e-4
            def fun(x):
                return np.sum(x > 1e-5), np.max(x)

            cka = np.array([fun(err[:, k]) for k in range(err.shape[1])])
            ck = np.all((cka[:, 0] < 15) & (cka[:, 1] < 1e-4))

            print(('\t\tTEST SIM {0:2d} : crit={1}'+\
                                ' err={2:3.3e} warmup={3}').format(\
                                i+1, ck, np.max(err), warmup_ideal))

            self.assertTrue(ck)


    def test_gr2m_calib(self):
        ''' Test gr2m calibration '''
        i = 0

        # Get data
        data = testdata.read('GR2M_timeseries_{0:02d}.csv'.format(i+1), \
                                source='output', has_dates=False)
        params = testdata.read('GR2M_params_{0:02d}.csv'.format(i+1), \
                                source='output', has_dates=False)

        inputs = np.ascontiguousarray(\
                        data.loc[:, ['Precip', 'PotEvap']], \
                        np.float64)

        gr = GR2M()
        gr.allocate(inputs)

        # Calibration object
        calib = CalibrationGR2M(objfun=ObjFunSSE(), nparamslib=1000)

        # Sample parameters
        nsamples = 10
        samples = calib.paramslib[:nsamples]

        # loop through parameters
        for i, expected in enumerate(samples):

            # Generate obs
            gr.params.values = expected
            gr.initialise()
            gr.run()

            # Produce theoretical observation
            # with error corruption
            err = np.random.uniform(-1, 1, gr.outputs.shape[0]) * 1e-3
            obs = gr.outputs[:,0].copy()+err

            # Calibrate
            final, ofun, _, _ = calib.workflow(obs, inputs, \
                                        maxfun=100000, ftol=1e-8)

            # Test
            err = np.abs(final-expected)
            imax = np.argmax(err)
            ck = np.allclose(expected, final, rtol=5e-2, atol=1e-3)
            print(('\t\tTEST CALIB {0:02d} : PASSED?{1:5}'+\
                        ' err[X{2}] = {3:3.3e}').format(\
                        i+1, str(ck), imax+1, err[imax]))

            self.assertTrue(ck)


    def test_gr2m_calib_fixed(self):
        ''' Test GR2M calibration with fixed parameters '''
        i = 0

        # Get data
        data = testdata.read('GR2M_timeseries_{0:02d}.csv'.format(i+1), \
                                source='output', has_dates=False)
        expected = testdata.read('GR2M_params_{0:02d}.csv'.format(i+1), \
                                source='output', has_dates=False)
        expected = expected.values[:, 0]

        inputs = np.ascontiguousarray(\
                        data.loc[:, ['Precip', 'PotEvap']], \
                        np.float64)

        gr = GR2M()
        gr.allocate(inputs, 1)

        # Calibration object
        calib1 = CalibrationGR2M(objfun=ObjFunSSE())
        calib2 = CalibrationGR2M(objfun=ObjFunSSE(), fixed={'X1':200})

        # Generate obs
        gr.params.values = expected
        gr.initialise()
        gr.run()

        # Produce theoretical observation
        # with error corruption
        err = np.random.uniform(-1, 1, gr.outputs.shape[0]) * 1e-3
        obs = gr.outputs[:,0].copy()+err

        # Calibrate
        final1, ofun1, _, _ = calib1.workflow(obs, inputs, \
                                    maxfun=100000, ftol=1e-8)
        final2, ofun2, _, _ = calib2.workflow(obs, inputs, \
                                    maxfun=100000, ftol=1e-8)

        self.assertTrue(np.allclose(final1, expected, atol=1e-1))
        self.assertTrue(np.allclose(final2[0], 200))


if __name__ == '__main__':
    unittest.main()
