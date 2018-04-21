import os
import re
import unittest

import time

import numpy as np

from pygme.calibration import ObjFunSSE
from pygme.models.gr6j import GR6J, CalibrationGR6J

import c_pygme_models_utils
UHEPS = c_pygme_models_utils.uh_getuheps()


class GR6JTestCases(unittest.TestCase):


    def setUp(self):
        print('\t=> GR6JTestCase')
        filename = os.path.abspath(__file__)
        self.FHERE = os.path.dirname(filename)


    def test_print(self):
        gr = GR6J()
        str_gr = '%s' % gr


    def test_error1(self):
        gr = GR6J()
        try:
            gr.allocate(np.random.uniform(0, 1, (200, 2)), 30)
        except ValueError as  err:
            self.assertTrue(str(err).startswith('model GR6J: Expected noutputs'))
        else:
            raise ValueError('Problem with error handling')


    def test_error2(self):
        gr = GR6J()
        inputs = np.random.uniform(size=(20, 3))
        try:
            gr.allocate(inputs, 5)
            gr.initialise()
        except ValueError as  err:
            self.assertTrue(str(err).startswith('model GR6J: Expected 2 inputs'))
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


    def test_run_against_data(self):
        ''' Compare GR6J simulation with test data '''
        warmup = 365 * 5
        gr = GR6J()

        for i in range(20):

            # Get inputs
            fts = os.path.join(self.FHERE, 'data_gr6j', \
                    'GR6J_timeseries_{0:02d}.csv'.format(i+1))
            data = np.loadtxt(fts, delimiter=',', skiprows=1)
            inputs = np.ascontiguousarray(data[:, [1, 0]], np.float64)

            # Get parameters
            fp = os.path.join(self.FHERE, 'data_gr6j', \
                    'GR6J_params_{0:02d}.csv'.format(i+1))
            params = np.loadtxt(fp, delimiter=',', skiprows=1)

            # Run gr6j
            gr.allocate(inputs, 11)
            t0 = time.time()
            gr.params.values = params

            # .. initiase to same values than IRSTEA run ...
            gr.initialise(states=data[0, [2, 10, 17]])

            gr.run()
            qsim1 = gr.outputs[:,0].copy()
            t1 = time.time()
            dta = 1000 * (t1-t0)
            dta /= len(qsim1)*365.25

            # Compare
            idx = np.arange(inputs.shape[0]) > warmup
            expected = data[idx, -1]

            err = np.abs(qsim1[idx] - expected)
            err_thresh = 5e-2
            ck = np.max(err) < err_thresh

            if not ck:
                print(('\t\tTEST %2d : max abs err = '
                    '%0.5f > %0.5f, runtime = %0.5fms/yr\n') % (i+1, \
                    np.max(err), err_thresh, dta))
            else:
                print(('\t\tTEST %2d : max abs err = %0.5f\n\t\t\truntime =' +
                        ' %0.5fms/yr\n') % ( \
                    i+1, np.max(err), dta))

            self.assertTrue(ck)


    def test_calibrate_against_itself(self):
        ''' Calibrate GR6J against a simulation with known parameters '''
        return
        gr = GR6J()
        warmup = 365*6
        calib = CalibrationGR6J(objfun=ObjFunSSE())

        fp = '{0}/data/GR6J_params.csv'.format(self.FHERE)
        params = np.loadtxt(fp, delimiter=',')

        for i in range(params.shape[0]):

            fts = '{0}/data/GR6J_timeseries_{1:02d}.csv'.format( \
                    self.FHERE, i+1)
            data = np.loadtxt(fts, delimiter=',')
            inputs = np.ascontiguousarray(data[:, [1, 2]], np.float64)

            # Run gr first
            expected = params[i, [2, 0, 1, 3]]
            gr = calib.model
            gr.allocate(inputs, 1)
            gr.params.values = expected
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
            err = np.abs(final-expected)
            ck = np.max(err) < 1e-2

            print(('\t\tTEST CALIB {0:02d} : max abs err = {1:3.3e}'+\
                    ' dt={2:3.3e} sec/yr').format(\
                        i+1, np.max(err), dt))

            self.assertTrue(ck)

