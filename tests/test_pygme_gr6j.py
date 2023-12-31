import os
import re
import cProfile
import unittest
import math
from itertools import product as prod

import time

import numpy as np

from pygme.calibration import ObjFunSSE
from pygme.models.gr6j import GR6J, CalibrationGR6J
from pygme.models.gr4j import compute_PmEm, gr4j_X1_initial

import testdata

import c_pygme_models_utils
UHEPS = c_pygme_models_utils.uh_getuheps()

PROFILE = False

class GR6JTestCase(unittest.TestCase):


    def setUp(self):
        print('\t=> GR6JTestCase')
        filename = os.path.abspath(__file__)
        self.ftest = os.path.dirname(filename)


    def test_print(self):
        gr = GR6J()
        str_gr = '%s' % gr


    def test_error1(self):
        gr = GR6J()
        try:
            gr.allocate(np.random.uniform(0, 1, (200, 2)), 30)
        except ValueError as  err:
            self.assertTrue(str(err).startswith(\
                            'model GR6J: Expected noutputs'))
        else:
            raise ValueError('Problem with error handling')


    def test_error2(self):
        gr = GR6J()
        inputs = np.random.uniform(size=(20, 3))
        try:
            gr.allocate(inputs, 5)
            gr.initialise()
        except ValueError as  err:
            self.assertTrue(str(err).startswith(\
                    'model GR6J: Expected 2 inputs'))
        else:
            raise ValueError('Problem with error handling')


    def test_version(self):
        ''' Test GR6J version '''
        gr = GR6J()
        gr.allocate(np.zeros((10, 2)))

        for version in  [0, 1]:
            gr.config.version = version
            gr.run()

        gr.config.version = 2
        try:
            gr.run()
        except ValueError as err:
            self.assertTrue(str(err).startswith(\
                        'c_pygme_models_hydromodels.gr6j_run '+\
                        'returns'))
        else:
            raise ValueError('Problem with error handling')


    def test_uh(self):
        ''' Test GR6J UH '''
        gr = GR6J()
        gr.allocate(np.zeros((10, 2)))

        for x4 in np.linspace(0, 50, 100):
            # Set parameters
            gr.X1 = 400
            gr.X2 = -1
            gr.X3 = 50
            gr.X4 = x4

            ord1 = gr.params.uhs[0][1].ord
            ord2 = gr.params.uhs[1][1].ord

            ck = abs(np.sum(ord1)-1) < UHEPS * 2
            self.assertTrue(ck)

            ck = abs(np.sum(ord2)-1) < UHEPS * 2
            self.assertTrue(ck)


    def test_run_dimensions(self):
        ''' Allocate GR6J '''
        gr = GR6J()
        nval = 1000

        p = np.exp(np.random.normal(0, 2, size=nval))
        pe = np.ones(nval) * 5.

        inputs = np.array([p, pe]).T
        gr.allocate(inputs, 9)
        gr.initialise()
        gr.run()

        out = gr.outputs
        ck = out.shape == (nval, 9)
        self.assertTrue(ck)


    def test_initialisation(self):
        ''' Test GR6J initialisation '''
        gr = GR6J()
        warmup = 365*6

        for i in range(20):
            print('Running GR6J initialisation test {0}/{1}'.format(i+1, 20))
            data = testdata.read('GR6J_timeseries_{0:02d}.csv'.format(i+1), \
                                    source='output', has_dates=False)
            inputs = np.ascontiguousarray(\
                            data.loc[:, ['Precip', 'PotEvap']], \
                            np.float64)

            Pm, Em = compute_PmEm(inputs[:, 0], inputs[:, 1])
            Q0 = np.nanmean(inputs[:, -1])

            for itest, (X1, X3, X6) in enumerate(prod(np.logspace(0, 4, 5), \
                        np.logspace(0, 4, 5), np.logspace(0, 2, 5))):
                gr.params.X1 = X1
                gr.params.X3 = X3
                gr.params.X6 = X6

                # Wrapper function for profiling
                def profilewrap():
                    gr.initialise_fromdata(Pm, Em, Q0)

                # Run profiler
                if PROFILE and i==0 and itest == 50:
                    pstats = os.path.join(self.ftest, \
                                    'gr6j_initdata{0:02d}.pstats'.format(i))
                    prof = cProfile.runctx('profilewrap()', globals(), \
                                locals(), filename=pstats)
                else:
                    profilewrap()

                ini = gr4j_X1_initial(Pm, Em, X1)

                self.assertTrue(np.isclose(gr.states.values[0], ini*X1))
                self.assertTrue(np.isclose(gr.states.values[1], 0.3*X3))
                self.assertTrue(np.isclose(gr.states.values[2], \
                                        X6*math.log(math.exp(Q0/X6)-1)))

                gr.initialise_fromdata()
                self.assertTrue(np.isclose(gr.states.values[0], 0.5*X1))
                self.assertTrue(np.isclose(gr.states.values[1], 0.3*X3))
                self.assertTrue(np.isclose(gr.states.values[2], \
                                        X6*math.log(math.exp(1e-3/X6)-1)))


    def test_run_against_data(self):
        ''' Compare GR6J simulation with test data '''
        warmup = 365 * 10
        gr = GR6J()

        for i in range(20):
            # Get data
            data = testdata.read('GR6J_timeseries_{0:02d}.csv'.format(i+1), \
                                    source='output', has_dates=False)
            params = testdata.read('GR6J_params_{0:02d}.csv'.format(i+1), \
                                    source='output', has_dates=False)
            params = params.values[:, 0]

            inputs = np.ascontiguousarray(\
                            data.loc[:, ['Precip', 'PotEvap']], \
                            np.float64)

            # Run gr6j
            gr.allocate(inputs, 13)
            gr.params.values = params

            # .. initiase to same values than IRSTEA run ...
            # Estimate initial states based on first two state values
            s0 = data.loc[0, ['Prod', 'Rout', 'Exp']]
            s1 = data.loc[1, ['Prod', 'Rout', 'Exp']]
            sini = 2*s0-s1
            gr.initialise(states=sini)
            gr.run()

            # Collect Q, QD, QR, QExp
            sim = gr.outputs[:, [0, 7, 8, 10]].copy()
            expstore = gr.outputs[:, 3]

            # Compare
            idx = np.arange(inputs.shape[0]) > warmup
            expected = data.loc[:, ['Qsim', 'QD', 'QR', 'QRExp']].values
            err = np.abs(sim[idx, :] - expected[idx, :])

            # Sensitivity to initial conditionos
            s1 = [0, 0, -100-gr.params.X6]
            s2 = [gr.params.X1, gr.params.X3, -gr.params.X6]
            warmup_ideal, sim0, sim1 = gr.inisens(s1, s2, \
                                eps=1e-3, ignore_error=True)

            # Special criteria
            # 5 values with difference greater than 1e-5
            # max diff lower than 5e-4
            def fun(x):
                return np.sum(x > 1e-5), np.max(x)

            cka = np.array([fun(err[:, k]) for k in range(err.shape[1])])
            ck = np.all((cka[:, 0] < 5) & (cka[:, 1] < 1e-4))

            print(('\t\tTEST SIM {0:2d} : '+#
                        'crit={1} err={2:3.3e} warmup={3}').format(\
                        i+1, ck, np.max(err), warmup_ideal))

            self.assertTrue(ck)


    def test_calibrate_against_itself(self):
        ''' Calibrate GR6J against a simulation with known parameters '''
        gr = GR6J()
        warmup = 365*6

        for i in range(3):
            data = testdata.read('GR6J_timeseries_{0:02d}.csv'.format(i+1), \
                                    source='output', has_dates=False)
            params = testdata.read('GR6J_params_{0:02d}.csv'.format(i+1), \
                                    source='output', has_dates=False)
            params = params.values[:, 0]

            inputs = np.ascontiguousarray(\
                            data.loc[:, ['Precip', 'PotEvap']], \
                            np.float64)
            # Run gr first
            gr.allocate(inputs, 1)
            gr.params.values = params
            gr.initialise()
            gr.run()

            # Calibrate
            Pm, Em = compute_PmEm(inputs[:, 0], inputs[:, 1])
            calib = CalibrationGR6J(objfun=ObjFunSSE(), Pm=Pm, Em=Em)

            noise = np.random.uniform(-1, 1, gr.outputs.shape[0]) * 1e-4
            sim0 = gr.outputs[:, 0].copy()
            obs = np.maximum(sim0+noise, 0.)

            t0 = time.time()

            # Wrapper function for profiling
            def profilewrap(outputs):
                final, ofun, _, _ = calib.workflow(obs, inputs, \
                                            maxfun=100000, ftol=1e-8)
                outputs.append(final)
                outputs.append(ofun)

            # Run profiler
            if PROFILE and i == 0:
                pstats = os.path.join(self.ftest, \
                                'gr6j_calib{0:02d}.pstats'.format(i))
                outputs = []
                prof = cProfile.runctx('profilewrap(outputs)', globals(), \
                            locals(), filename=pstats)
                final = outputs[0]
                ofun = outputs[1]
            else:
                outputs = []
                profilewrap(outputs)
                final = outputs[0]
                ofun = outputs[1]

            t1 = time.time()
            dt = (t1-t0)/len(obs)*365.25

            # Test error on parameters
            err = np.abs(final-params)
            ck1 = np.max(err) < 1e-2

            idx = np.arange(len(sim0)) > warmup
            sim = gr.outputs[idx, 0]
            errs = np.abs(sim-sim0[idx])
            ck2 = np.max(errs) < 5e-3

            print(('\t\tTEST CALIB {0:02d} : max abs err = {1:3.3e}'+\
                    ' dt={2:3.3e} sec/yr').format(\
                        i+1, np.max(err), dt))

            self.assertTrue(ck1 or ck2)


if __name__ == "__main__":
    unittest.main()
