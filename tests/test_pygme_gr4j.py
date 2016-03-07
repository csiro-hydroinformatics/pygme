import os
import re
import unittest

import time

import numpy as np

from pygme import calibration
from pygme.data import Matrix
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
        except ValueError as  e:
            pass
        self.assertTrue(str(e).startswith('With model gr4j, Number of outputs'))


    def test_error2(self):
        gr = GR4J()
        inputs = np.random.uniform(size=(20, 3))
        try:
            gr.allocate(inputs, 5)
            gr.initialise()
        except ValueError as  e:
            pass
        self.assertTrue(str(e).startswith('With model gr4j, Number of inputs'))


    def test_uh(self):
        gr = GR4J()
        gr.allocate(np.zeros((10, 2)))
        for x4 in np.linspace(0, 50, 100):
            gr.params = [400, -1, 50, x4]

            ck = abs(np.sum(gr.uh)-2) < UHEPS * 2
            self.assertTrue(ck)


    def test_run1(self):
        gr = GR4J()
        nval = 1000

        p = np.exp(np.random.normal(0, 2, size=nval))
        pe = np.ones(nval) * 5.

        inputs = np.array([p, pe]).T
        gr.allocate(inputs, 9)
        gr.initialise()
        gr.reset('params')
        gr.run()

        out = gr.outputs
        ck = out.shape == (nval, 9)
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


            # Run gr4j [block]
            gr.allocate(inputs)
            t0 = time.time()
            gr.params = params[i, [2, 0, 1, 3]]
            gr.initialise()
            gr.run_as_block = True
            gr.run()
            qsim1 = gr.outputs[:,0].copy()
            t1 = time.time()
            dta1 = 1000 * (t1-t0)
            dta1 /= len(qsim1)/365.25

            # Run gr4j [timestep]
            gr._outputs.reset()
            t0 = time.time()
            gr.initialise()
            gr.run_as_block = False
            gr.run()
            qsim2 = gr.outputs[:,0].copy()
            t1 = time.time()
            dta2 = 1000 * (t1-t0)
            dta2 /= len(qsim2)/365.25

            # Compare
            idx = np.arange(inputs.shape[0]) > warmup
            expected = data[idx, 4]

            err = np.abs(qsim1[idx] - expected)
            err_thresh = 5e-2
            ck1 = np.max(err) < err_thresh

            err = np.abs(qsim2[idx] - qsim1[idx])
            err_thresh = 5e-2
            ck2 = np.max(err) < err_thresh

            if not (ck1 and ck2):
                print(('\t\tTEST %2d : max abs err = '
                    '%0.5f < %0.5f ? %s ~ %0.5fms/yr\n') % (i+1, \
                    np.max(err), err_thresh, ck1, ck2))
            else:
                print(('\t\tTEST %2d : max abs err = %0.5f\n\t\t\truntime :' +
                        ' %0.5fms/yr [block] / %0.5fms/yr [ts]\n') % ( \
                    i+1, np.max(err), dta1, dta2))

            self.assertTrue(ck1)
            self.assertTrue(ck2)


    def test_calibrate(self):
        gr = GR4J()
        warmup = 365*6

        calib = CalibrationGR4J()
        calib.errfun = calibration.ssqe_bias


        fp = '{0}/data/GR4J_params.csv'.format(self.FHERE)
        params = np.loadtxt(fp, delimiter=',')

        for i in range(params.shape[0]):

            fts = '{0}/data/GR4J_timeseries_{1:02d}.csv'.format( \
                    self.FHERE, i+1)
            data = np.loadtxt(fts, delimiter=',')
            inputs = Matrix.from_data('inputs',
                    np.ascontiguousarray(data[:, [1, 2]], np.float64))

            # Run gr first
            params_expected = params[i, [2, 0, 1, 3]]
            gr = calib.model
            gr.allocate(inputs, 1)
            gr.params = params_expected
            gr.initialise()
            gr.run()
            obs = Matrix.from_data('obs', gr.outputs[:,0].copy())

            # Calibrate
            calib.setup(obs, inputs)

            start, explo, explo_ofun = calib.explore()
            ieval1 = calib.ieval

            final, _, _ = calib.fit(start, ftol=1e-8)
            ieval2 = calib.ieval - ieval1

            err = np.abs(calib.model.params - params_expected)
            ck = np.max(err) < 1e-8

            print(('\t\tTEST CALIB {0:02d} : max abs err = {1:3.3e}' +
                    ' neval= {2} + {3}').format( \
                        i+1, np.max(err), ieval1, ieval2))

            self.assertTrue(ck)

