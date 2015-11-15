import os
import re
import unittest

from timeit import Timer
import time

import numpy as np

from useme.models.gr2m import GR2M, CalibrationGR2M
from useme import calibration

class GR2MTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> GR2MTestCase')
        filename = os.path.abspath(__file__)
        self.FHERE = os.path.dirname(filename)


    def test_print(self):
        gr = GR2M()
        str_gr = '%s' % gr


    def test_sample(self):
        nsamples = 100
        obs = np.zeros(10)
        inputs = np.zeros((10, 2))
        calib = CalibrationGR2M()
        samples = calib.sample(nsamples)
        self.assertTrue(samples.shape == (nsamples, 2))


    def test_gr2m_dumb(self):
        gr = GR2M()
        nval = 100
        gr.allocate(nval, 9)
        gr.params.data = [400, 0.9]
        gr.initialise()

        p = np.exp(np.random.normal(0, 2, size=nval))
        pe = np.ones(nval) * 5.
        gr.inputs.data = np.concatenate([p[:,None], pe[:, None]], axis=1)

        gr.run()

        cols1 = gr.outputs.names

        cols2 = ['Q[mm/m]', 'Ech[mm/m]', 
           'P1[mm/m]', 'P2[mm/m]', 'P3[mm/m]',
           'R1[mm/m]', 'R2[mm/m]', 'S[mm]', 'R[mm]']

        ck = np.all(cols1 == cols2)
        self.assertTrue(ck)
 

    def test_gr2m_irstea(self):
        fd = '{0}/data/GR2M_timeseries.csv'.format(self.FHERE)
        data = np.loadtxt(fd, delimiter=',', skiprows=1)
        inputs = np.ascontiguousarray(data[:, :2])

        params = [650.7, 0.8]

        # Run
        gr = GR2M()
        gr.allocate(len(inputs), 9)
        gr.params.data = params
        gr.inputs.data = inputs
        gr.initialise()
        gr.run()
        out = gr.outputs.data

        # Test
        warmup = 30
        res = out[warmup:,]
        expected = data[warmup:, [13, 10, 4, 7, 8, 9, 11, 6, 12]]

        for i in range(res.shape[1]):
            err = np.abs(res[:,i] - expected[:,i])

            err_thresh = 4e-1
            if not i in [0, 1, 6]:
                err_thresh = 1e-2 * np.min(np.abs(expected[expected[:,i]!=0.,i]))
            ck = np.max(err) < err_thresh

            if not ck:
                print('\tVAR[%d] : max abs err = %0.5f < %0.5f ? %s' % ( \
                        i, np.max(err), err_thresh, ck)) 

            self.assertTrue(ck)
 

    def test_gr2m_irstea_calib(self):
        fd = '{0}/data/GR2M_timeseries.csv'.format(self.FHERE)
        data = np.loadtxt(fd, delimiter=',', skiprows=1)
        inputs = np.ascontiguousarray(data[:, :2])

        calparams_expected = [650.7, 0.8]

        gr = GR2M()
        gr.allocate(len(inputs), 1)
        gr.inputs.data = inputs

        # Parameter samples
        nsamples = 50
        calib = CalibrationGR2M()
        calib.errfun = calibration.ssqe_bias
        samples = calib.sample(nsamples)

        idx_cal = np.arange(12, len(inputs))
        
        # loop through parameters
        for i in range(nsamples):

            # Generate obs
            gr.params.data = np.exp(samples[i, :])
            expected = gr.params.data.copy()
            gr.initialise()
            gr.run()
            obs = gr.outputs.data[:,0].copy()

            # Calibrate
            calib.setup(obs, inputs)
            calib.idx_cal = idx_cal
                        
            calparams_ini, _, _ = calib.explore()
            calparams_final, _, _ = calib.fit(calparams_ini, iprint=0)

            err = np.abs(gr.params.data-expected)
            ck = np.max(err) < 1e-5

            self.assertTrue(ck)

