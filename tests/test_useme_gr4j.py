import os
import re
import unittest

import time

import numpy as np

from useme import calibration
from useme.models.gr4j import GR4J, CalibrationGR4J


import c_useme_models_utils
UHEPS = c_useme_models_utils.uh_getuheps()


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
            gr.allocate(20, 30)
        except ValueError as  e:
            pass
        self.assertTrue(e.message.startswith('Too many outputs'))


    def test_error2(self):
        gr = GR4J()
        gr.allocate(20, 5)
        gr.initialise()
        try:
            gr.inputs.data = np.random.uniform(size=(20, 3))
        except ValueError as  e:
            pass
        self.assertTrue(e.message.startswith('inputs matrix'))


    def test_sample(self):
        nsamples = 100
        obs = np.zeros(10)
        inputs = np.zeros((10, 2))
        calib = CalibrationGR4J()
        samples = calib.sample(nsamples)
        self.assertTrue(samples.shape == (nsamples, 4))


    def test_uh(self):
        gr = GR4J()
        for x4 in np.linspace(0, 50, 100):
            gr.params.data = [400, -1, 50, x4]

            ck = abs(np.sum(gr.uh.data)-2) < UHEPS * 2
            self.assertTrue(ck)


    def test_run1(self):
        gr = GR4J()
        nval = 1000
        gr.allocate(nval, 9)

        p = np.exp(np.random.normal(0, 2, size=nval))
        pe = np.ones(nval) * 5.
        gr.inputs.data = np.concatenate([p[:,None], pe[:, None]], axis=1)

        gr.initialise()
        gr.run()

        out = gr.outputs.data
        cs = gr.outputs.names

        cols = ['Q[mm/d]', 'Ech[mm/d]',
           'E[mm/d]', 'Pr[mm/d]',
           'Qd[mm/d]', 'Qr[mm/d]',
           'Perc[mm/d]', 'S[mm]', 'R[mm]']

        ck = np.all(list(cs) == cols)
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
            inputs = data[:, [1, 2]]
            inputs = np.ascontiguousarray(inputs, np.float64)

            # Run gr4j
            gr.allocate(len(inputs), 1)
            gr.inputs.data = inputs
            t0 = time.time()

            gr.params.data = params[i, [2, 0, 1, 3]]
            gr.initialise()
            gr.run()
            qsim = gr.outputs.data[:,0]

            t1 = time.time()
            dta = 1000 * (t1-t0)
            dta /= len(qsim)/365.25

            qsim = gr.outputs.data[:, 0]

            # Compare
            idx = np.arange(len(inputs)) > warmup
            expected = data[idx, 4]

            err = np.abs(qsim[idx] - expected)
            err_thresh = 5e-2
            ck = np.max(err) < err_thresh

            if not ck:
                print(('\t\tTEST %2d : max abs err = '
                    '%0.5f < %0.5f ? %s ~ %0.5fms/yr') % (i+1, \
                    np.max(err), err_thresh, ck, dta))
            else:
                print('\t\tTEST %2d : max abs err = %0.5f ~ %0.5fms/yr' % ( \
                    i+1, np.max(err), dta))

            self.assertTrue(ck)


    def test_calibrate(self):
        gr = GR4J()
        warmup = 365*6

        calib = CalibrationGR4J()
        calib.errfun = calibration.ssqe_bias

        warmup = 365 * 5

        fp = '{0}/data/GR4J_params.csv'.format(self.FHERE)
        params = np.loadtxt(fp, delimiter=',')

        for i in range(params.shape[0]):

            fts = '{0}/data/GR4J_timeseries_{1:02d}.csv'.format( \
                    self.FHERE, i+1)
            data = np.loadtxt(fts, delimiter=',')
            inputs = data[:, [1, 2]]
            inputs = np.ascontiguousarray(inputs, np.float64)
            nval = inputs.shape[0]
            idx_cal = np.arange(len(inputs))>=warmup
            idx_cal = np.where(idx_cal)[0]
            calib.setup(inputs[:,0]*0., inputs)

            # Run gr first
            params_expected = params[i, [2, 0, 1, 3]]
            gr = calib.model
            gr.params.data = params_expected
            gr.initialise()
            gr.inputs.data = inputs
            gr.run()
            obs = gr.outputs.data[:,0].copy()

            # Calibrate
            calib.observations.data = obs
            calib.idx_cal = idx_cal
            
            ini, explo, explo_ofun = calib.explore()
            ieval1 = calib.ieval

            final, _, _ = calib.fit(ini)
            ieval2 = calib.ieval

            err = np.abs(calib.model.params.data  \
                    - params_expected)
            ck = np.max(err) < 1e-8

            print('\t\tTEST CALIB {0:02d} : max abs err = {1:3.3e} neval={2}/{3}'.format( \
                    i+1, np.max(err), ieval1, ieval2))

            self.assertTrue(ck)

