import os
import re
import unittest

import time

import numpy as np

from pygme.calibration import ObjFunSSE
from pygme.models.hbv import HBV, CalibrationHBV

import c_pygme_models_utils
UHEPS = c_pygme_models_utils.uh_getuheps()


class HBVTestCases(unittest.TestCase):


    def setUp(self):
        print('\t=> HBVTestCase')
        filename = os.path.abspath(__file__)
        self.ftest = os.path.dirname(filename)


    def test_print(self):
        gr = HBV()
        str_gr = '%s' % gr


    def test_error1(self):
        gr = HBV()
        try:
            gr.allocate(np.random.uniform(0, 1, (200, 2)), 30)
        except ValueError as  err:
            self.assertTrue(str(err).startswith('model HBV: Expected noutputs'))
        else:
            raise ValueError('Problem with error handling')


    def test_error2(self):
        gr = HBV()
        inputs = np.random.uniform(size=(20, 3))
        try:
            gr.allocate(inputs, 5)
            gr.initialise()
        except ValueError as  err:
            self.assertTrue(str(err).startswith('model HBV: Expected 2 inputs'))
        else:
            raise ValueError('Problem with error handling')


    def test_uh(self):
        ''' Test HBV UH '''
        gr = HBV()
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
        ''' Allocate HBV '''
        gr = HBV()
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


    def test_run_against_data(self):
        ''' Compare HBV simulation with test data '''
        warmup = 365 * 10
        gr = HBV()

        #for i in range(13, 14):
        for i in range(20):
            # Get inputs
            fts = os.path.join(self.ftest, 'output_data', \
                    'HBV_timeseries_{0:02d}.csv'.format(i+1))
            data = np.loadtxt(fts, delimiter=',', skiprows=1)
            inputs = np.ascontiguousarray(data[:, [1, 0]], np.float64)

            # Get parameters
            fp = os.path.join(self.ftest, 'output_data', \
                    'HBV_params_{0:02d}.csv'.format(i+1))
            params = np.loadtxt(fp, delimiter=',', skiprows=1)

            # Run hbv
            gr.allocate(inputs, 11)
            gr.params.values = params

            # .. initiase to same values than IRSTEA run ...
            # Estimate initial states based on first two state values
            s0 = data[0, [2, 10, 17]]
            s1 = data[1, [2, 10, 17]]
            sini = 2*s0-s1
            gr.initialise(states=sini)

            gr.run()
            sim = gr.outputs[:, [0, 4, 5, 7]].copy()

            # Compare
            idx = np.arange(inputs.shape[0]) > warmup
            expected = data[:, [19, 18, 15, 16]]

            err = np.abs(sim[idx, :] - expected[idx, :])

            # Sensitivity to initial conditionos
            s1 = [0]*3
            s2 = [gr.params.X1, gr.params.X3, gr.params.X6]
            warmup_ideal = gr.inisens(s1, s2)

            # Special criteria
            # 5 values with difference greater than 1e-5
            # max diff lower than 5e-4
            def fun(x):
                return np.sum(x > 1e-5), np.max(x)

            cka = np.array([fun(err[:, k]) for k in range(err.shape[1])])
            ck = np.all((cka[:, 0] < 5) & (cka[:, 1] < 1e-4))

            print('\t\tTEST SIM {0:2d} : crit={1} err={2:3.3e} warmup={3}'.format(\
                                        i+1, ck, np.max(err), warmup_ideal))

            self.assertTrue(ck)


    def test_calibrate_against_itself(self):
        ''' Calibrate HBV against a simulation with known parameters '''
        gr = HBV()
        warmup = 365*6
        calib = CalibrationHBV(objfun=ObjFunSSE())

        for i in range(8, 9):
            # Get inputs
            fts = os.path.join(self.ftest, 'output_data', \
                    'HBV_timeseries_{0:02d}.csv'.format(i+1))
            data = np.loadtxt(fts, delimiter=',', skiprows=1)
            inputs = np.ascontiguousarray(data[:, [1, 0]], np.float64)

            # Get parameters
            fp = os.path.join(self.ftest, 'output_data', \
                    'HBV_params_{0:02d}.csv'.format(i+1))
            params = np.loadtxt(fp, delimiter=',', skiprows=1)

            # Run gr first
            gr = calib.model
            gr.allocate(inputs, 1)
            gr.params.values = params
            gr.initialise()
            gr.run()

            # Calibrate
            err = np.random.uniform(-1, 1, gr.outputs.shape[0]) * 1e-4
            sim0 = gr.outputs[:, 0].copy()
            obs = sim0+err

            t0 = time.time()
            final, ofun, _ = calib.workflow(obs, inputs, \
                                        maxfun=100000, ftol=1e-8)
            t1 = time.time()
            dt = (t1-t0)/len(obs)*365.25

            # Test error on parameters
            err = np.abs(final-params)
            ck1 = np.max(err) < 1e-2

            idx = np.arange(len(sim0)) > warmup
            errs = np.abs(gr.outputs[idx, 0]-sim0[idx])
            ck2 = np.max(errs) < 1e-3

            print(('\t\tTEST CALIB {0:02d} : max abs err = {1:3.3e}'+\
                    ' dt={2:3.3e} sec/yr').format(\
                        i+1, np.max(err), dt))

            if not (ck1 or ck2):
                import pdb; pdb.set_trace()

            self.assertTrue(ck)


