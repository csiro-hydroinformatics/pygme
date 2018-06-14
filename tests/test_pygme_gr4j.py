import os
import re
import zipfile
import cProfile
import unittest
from itertools import product as prod

import time

import numpy as np

from pygme.calibration import ObjFunSSE
from pygme.models.gr4j import GR4J, CalibrationGR4J
from pygme.models.gr4j import compute_PmEm, gr4j_X1_initial

import testdata

import c_pygme_models_hydromodels

import c_pygme_models_utils
UHEPS = c_pygme_models_utils.uh_getuheps()

PROFILE = True

class InitialTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> PmEmTestCase')
        filename = os.path.abspath(__file__)
        self.ftest = os.path.dirname(filename)

        # Check test data
        testdata.check_all()


    def test_PmEm(self):
        ''' Test Pm and Em '''

        for i in range(20):
            fts = '{0}/output_data/GR4J_timeseries_{1:02d}.csv'.format( \
                    self.ftest, i+1)
            data = np.loadtxt(fts, delimiter=',', skiprows=1)
            inputs = np.ascontiguousarray(data[:, [1, 0]], np.float64)

            Pm, Em = compute_PmEm(inputs[:, 0], inputs[:, 1])

            ts = inputs[:, 0] - inputs[:, 1]
            raine = np.maximum(ts, 0)
            idx = inputs[:, 0] >= inputs[:, 1]
            Pme = np.mean(raine[idx])
            self.assertTrue(np.isclose(Pm, Pme))
            evape = np.maximum(-ts, 0)
            Eme = np.mean(evape[~idx])
            self.assertTrue(np.isclose(Em, Eme))


    def test_initial(self):
        ''' Test initialisation of GR4J X1 '''

        # Test the case where Pm=0, Em=0
        for X1 in np.logspace(0, 5, 100):
            ini = gr4j_X1_initial(0., 0., X1)
            self.assertTrue(np.isclose(ini, 0.))

        # Objective function for initial condition
        def fun(ini, Pm, Em, X1):
            ratio = ini/2.25
            isq = 1./(1+ratio**4)**0.25
            f = (1-ini**2)*Pm-ini*(2-ini)*Em-X1*ini*(1-isq);
            return f

        # Loop over test catchments
        for i in range(20):
            # Data
            fts = '{0}/output_data/GR4J_timeseries_{1:02d}.csv'.format( \
                    self.ftest, i+1)
            data = np.loadtxt(fts, delimiter=',', skiprows=1)
            inputs = np.ascontiguousarray(data[:, [1, 0]], np.float64)
            Pm, Em = compute_PmEm(inputs[:, 0], inputs[:, 1])

            # Test of multiple X1 values
            for X1 in np.logspace(0, 5, 100):
                ini = gr4j_X1_initial(Pm, Em, X1)
                f = fun(ini, Pm, Em, X1)
                self.assertTrue(np.isclose(f, 0., rtol=0., atol=1e-3))


    def test_initial_error(self):
        ''' Test initialisation error '''

        try:
            ini = gr4j_X1_initial(-1, 1, 1)
        except ValueError as  err:
            self.assertTrue(str(err).startswith(\
                'c_pygme_models_hydromodels.gr4j_X1_initial'))
        else:
            raise ValueError('Problem with error handling')

        try:
            ini = gr4j_X1_initial(1, -1, 1)
        except ValueError as  err:
            self.assertTrue(str(err).startswith(\
                'c_pygme_models_hydromodels.gr4j_X1_initial'))
        else:
            raise ValueError('Problem with error handling')

        try:
            ini = gr4j_X1_initial(1, 1, -1)
        except ValueError as  err:
            self.assertTrue(str(err).startswith(\
                'c_pygme_models_hydromodels.gr4j_X1_initial'))
        else:
            raise ValueError('Problem with error handling')




class GR4JTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> GR4JTestCase')
        filename = os.path.abspath(__file__)
        self.ftest = os.path.dirname(filename)

        # Check test data
        testdata.check_all()

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
        gr.allocate(inputs, 11)
        gr.initialise()
        gr.run()

        out = gr.outputs
        ck = out.shape == (nval, 11)
        self.assertTrue(ck)


    def test_run_against_data(self):
        ''' Compare GR4J simulation with test data '''
        warmup = 365 * 5
        gr = GR4J()

        for i in range(20):
            fp = '{0}/output_data/GR4J_params_{1:02d}.csv'.format( \
                    self.ftest, i+1)
            params = np.loadtxt(fp, delimiter=',', skiprows=1)

            fts = '{0}/output_data/GR4J_timeseries_{1:02d}.csv'.format( \
                    self.ftest, i+1)
            data = np.loadtxt(fts, delimiter=',', skiprows=1)
            inputs = np.ascontiguousarray(data[:, [1, 0]], np.float64)

            # Run gr4j
            gr.allocate(inputs, 11)
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
            sim = gr.outputs[:, [0, 6, 7]].copy()
            expected = data[:, [17, 16, 15]]

            err = np.abs(sim[idx, :] - expected[idx, :])

            # Sensitivity to initial conditionos
            s1 = [0]*2
            s2 = [gr.params.X1, gr.params.X3]
            warmup_ideal, sim0, sim1 = gr.inisens(s1, s2)

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


    def test_initialisation(self):
        ''' Test GR4J initialisation '''
        gr = GR4J()
        warmup = 365*6

        for i in range(20):
            fts = '{0}/output_data/GR4J_timeseries_{1:02d}.csv'.format( \
                    self.ftest, i+1)
            data = np.loadtxt(fts, delimiter=',', skiprows=1)
            inputs = np.ascontiguousarray(data[:, [1, 0]], np.float64)

            Pm, Em = compute_PmEm(inputs[:, 0], inputs[:, 1])

            for X1, X3 in prod(np.logspace(0, 4, 5), np.logspace(0, 4, 5)):
                gr.params.X1 = X1
                gr.params.X3 = X3
                gr.initialise_fromdata(Pm, Em)
                ini = gr4j_X1_initial(Pm, Em, X1)
                self.assertTrue(np.isclose(gr.states.values[0], ini*X1))
                self.assertTrue(np.isclose(gr.states.values[1], 0.3*X3))

                gr.initialise_fromdata()
                self.assertTrue(np.isclose(gr.states.values[0], 0.5*X1))
                self.assertTrue(np.isclose(gr.states.values[1], 0.3*X3))


    def test_calibrate_against_itself(self):
        ''' Calibrate GR4J against a simulation with known parameters '''
        gr = GR4J()
        warmup = 365*6

        for i in range(20):
            fp = '{0}/output_data/GR4J_params_{1:02d}.csv'.format( \
                    self.ftest, i+1)
            params = np.loadtxt(fp, delimiter=',', skiprows=1)

            fts = '{0}/output_data/GR4J_timeseries_{1:02d}.csv'.format( \
                    self.ftest, i+1)
            data = np.loadtxt(fts, delimiter=',', skiprows=1)
            inputs = np.ascontiguousarray(data[:, [1, 0]], np.float64)
            Pm, Em = compute_PmEm(inputs[:, 0], inputs[:, 1])

            # Run gr first
            gr.allocate(inputs, 1)
            gr.params.values = params
            gr.initialise_fromdata(Pm, Em)
            gr.run()

            # Add error to outputs
            err = np.random.uniform(-1, 1, gr.outputs.shape[0]) * 1e-4
            obs = gr.outputs[:,0].copy()+err

            # Wrapper function for profiling
            t0 = time.time()
            calib = CalibrationGR4J(objfun=ObjFunSSE(), Pm=Pm, Em=Em)
            def profilewrap(outputs):
                final, ofun, _, _ = calib.workflow(obs, inputs, \
                                            maxfun=100000, ftol=1e-8)
                outputs.append(final)
                outputs.append(ofun)

            # Run profiler
            if PROFILE and i == 0:
                pstats = os.path.join(self.ftest, \
                                'gr4j_calib{0:02d}.pstats'.format(i))
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

        i = 10
        fp = '{0}/output_data/GR4J_params_{1:02d}.csv'.format( \
                self.ftest, i+1)
        expected = np.loadtxt(fp, delimiter=',', skiprows=1)

        fts = '{0}/output_data/GR4J_timeseries_{1:02d}.csv'.format( \
                self.ftest, i+1)
        data = np.loadtxt(fts, delimiter=',', skiprows=1)
        inputs = np.ascontiguousarray(data[:, [1, 0]], np.float64)

        # Run gr first
        gr = calib1.model
        gr.allocate(inputs, 1)
        gr.params.values = expected
        gr.initialise()
        gr.run()

        # Calibrate
        err = np.random.uniform(-1, 1, gr.outputs.shape[0]) * 1e-4
        obs = gr.outputs[:,0].copy()+err

        final1, ofun1, _, _ = calib1.workflow(obs, inputs, \
                                    maxfun=100000, ftol=1e-8)

        final2, ofun2, _, _ = calib2.workflow(obs, inputs, \
                                    maxfun=100000, ftol=1e-8)

        # Check one calibration works as expected
        self.assertTrue(np.allclose(final1, expected, atol=1e-1, rtol=0.))

        # Check the other one returns fixed parameters
        self.assertTrue(np.allclose(final2[[0, 3]], [1000, 10]))


if __name__ == "__main__":
    unittest.main()
