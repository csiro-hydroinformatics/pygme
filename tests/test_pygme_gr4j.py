import os
import re
import unittest

import time

import numpy as np

from pygme.calibration import ObjFunSSE
from pygme.models.gr4j import GR4J, CalibrationGR4J


import c_pygme_models_utils
UHEPS = c_pygme_models_utils.uh_getuheps()


class GR4JTestCases(unittest.TestCase):


    def setUp(self):
        print('\t=> GR4JTestCase')
        filename = os.path.abspath(__file__)
        self.FHERE = os.path.dirname(filename)


    def test_print(self):
        gr = GR4J()
        str_gr = '%s' % gr


    def test_error1(self):
        gr = GR4J()
        try:
            gr.allocate(np.random.uniform(0, 1, (200, 2)), 30)
        except ValueError as  err:
            self.assertTrue(str(err).startswith('model GR4J: Expected noutputs'))
        else:
            raise ValueError('Problem with error handling')


    def test_error2(self):
        gr = GR4J()
        inputs = np.random.uniform(size=(20, 3))
        try:
            gr.allocate(inputs, 5)
            gr.initialise()
        except ValueError as  err:
            self.assertTrue(str(err).startswith('model GR4J: Expected 2 inputs'))
        else:
            raise ValueError('Problem with error handling')


    def test_uh(self):
        ''' Test GR4J UH '''
        gr = GR4J()
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
        ''' Allocate GR4J '''
        gr = GR4J()
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
        ''' Compare GR4J simulation with test data '''
        warmup = 365 * 5
        gr = GR4J()

        for i in range(20):
            fp = '{0}/output_data/GR4J_params_{1:02d}.csv'.format( \
                    self.FHERE, i+1)
            params = np.loadtxt(fp, delimiter=',', skiprows=1)

            fts = '{0}/output_data/GR4J_timeseries_{1:02d}.csv'.format( \
                    self.FHERE, i+1)
            data = np.loadtxt(fts, delimiter=',', skiprows=1)
            inputs = np.ascontiguousarray(data[:, [1, 0]], np.float64)

            # Run gr4j
            gr.allocate(inputs, 9)
            gr.params.values = params

            # .. initiase to same values than IRSTEA run ...
            # Estimate initial states based on first two state values
            s0 = data[0, [2, 10]]
            s1 = data[1, [2, 10]]
            sini = 2*s0-s1
            gr.initialise(states=sini)

            gr.run()

            # Compare
            idx = np.arange(inputs.shape[0]) > warmup
            sim = gr.outputs[:, [0, 4, 5]].copy()
            expected = data[:, [17, 16, 15]]

            err = np.abs(sim[idx, :] - expected[idx, :])

            # Special criteria
            # 5 values with difference greater than 1e-5
            # max diff lower than 5e-4
            def fun(x):
                return np.sum(x > 1e-5), np.max(x)

            cka = np.array([fun(err[:, k]) for k in range(err.shape[1])])
            ck = np.all((cka[:, 0] < 5) & (cka[:, 1] < 1e-4))

            print('\t\tTEST {0} : crit={1} max abs err={2:3.3e}'.format(\
                                        i+1, ck, np.max(err)))

            self.assertTrue(ck)


    def test_calibrate_against_itself(self):
        ''' Calibrate GR4J against a simulation with known parameters '''
        gr = GR4J()
        warmup = 365*6
        calib = CalibrationGR4J(objfun=ObjFunSSE())

        for i in range(20):
            fp = '{0}/output_data/GR4J_params_{1:02d}.csv'.format( \
                    self.FHERE, i+1)
            params = np.loadtxt(fp, delimiter=',', skiprows=1)

            fts = '{0}/output_data/GR4J_timeseries_{1:02d}.csv'.format( \
                    self.FHERE, i+1)
            data = np.loadtxt(fts, delimiter=',', skiprows=1)
            inputs = np.ascontiguousarray(data[:, [1, 0]], np.float64)

            # Run gr first
            gr = calib.model
            gr.allocate(inputs, 1)
            gr.params.values = params
            gr.initialise()
            gr.run()

            # Calibrate
            err = np.random.uniform(-1, 1, gr.outputs.shape[0]) * 1e-4
            obs = gr.outputs[:,0].copy()+err

            t0 = time.time()
            final, ofun, _ = calib.workflow(obs, inputs, \
                                        maxfun=100000, ftol=1e-8)
            t1 = time.time()

            dt = (t1-t0)*1e-3/len(obs)*365.25

            # Test error on parameters
            err = np.abs(final-params)
            ck = np.max(err) < 1e-2

            print(('\t\tTEST CALIB {0:02d} : max abs err = {1:3.3e}'+\
                    ' dt={2:3.3e} sec/yr').format(\
                        i+1, np.max(err), dt))

            self.assertTrue(ck)

    def test_calibrate_fixed(self):
        ''' Calibrate GR4J with fixed parameter '''
        gr = GR4J()
        warmup = 365*6
        calib1 = CalibrationGR4J(objfun=ObjFunSSE())
        calib2 = CalibrationGR4J(objfun=ObjFunSSE(), \
                        fixed={'X1':1000, 'X4':10})

        fp = '{0}/data/GR4J_params.csv'.format(self.FHERE)
        params = np.loadtxt(fp, delimiter=',')

        i = 0
        fts = '{0}/data/GR4J_timeseries_{1:02d}.csv'.format( \
                self.FHERE, i+1)
        data = np.loadtxt(fts, delimiter=',')
        inputs = np.ascontiguousarray(data[:, [1, 2]], np.float64)

        # Run gr first
        expected = params[i, [2, 0, 1, 3]]
        gr = calib1.model
        gr.allocate(inputs, 1)
        gr.params.values = expected
        gr.initialise()
        gr.run()

        # Calibrate
        err = np.random.uniform(-1, 1, gr.outputs.shape[0]) * 1e-4
        obs = gr.outputs[:,0].copy()+err

        final1, ofun1, _ = calib1.workflow(obs, inputs, \
                                    maxfun=100000, ftol=1e-8)

        final2, ofun2, _ = calib2.workflow(obs, inputs, \
                                    maxfun=100000, ftol=1e-8)

        # Check one calibration works as expected
        self.assertTrue(np.allclose(final1, expected, atol=1e-1))

        # Check the other one returns fixed parameters
        self.assertTrue(np.allclose(final2[[0, 3]], [1000, 10]))

