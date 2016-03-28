import os, sys
import re
import unittest

from timeit import Timer
import time

import numpy as np
from pygme.models.gr2m import GR2M, CalibrationGR2M
from pygme import calibration
from pygme.data import Matrix

class GR2MTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> GR2MTestCase')
        filename = os.path.abspath(__file__)
        self.FHERE = os.path.dirname(filename)


    def test_print(self):
        gr = GR2M()
        str_gr = '%s' % gr


    def test_gr2m_dumb(self):
        gr = GR2M()

        nval = 100
        p = np.exp(np.random.normal(0, 2, size=nval))
        pe = np.ones(nval) * 5.
        inputs = np.concatenate([p[:,None], pe[:, None]], axis=1)
        gr.allocate(inputs, 9)

        gr.params = [400, 0.9]
        gr.initialise()

        gr.run()


    def test_gr2m_irstea(self):
        fd = '{0}/data/GR2M_timeseries.csv'.format(self.FHERE)
        data = np.loadtxt(fd, delimiter=',', skiprows=1)

        params = [650.7, 0.8]

        # Run
        gr = GR2M()
        inputs = np.ascontiguousarray(data[:, :2])
        gr.allocate(inputs, 9)
        gr.params = params
        gr.inputs = inputs
        gr.initialise()
        gr.run()
        out = gr.outputs

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

        calparams_expected = [650.7, 0.8]

        gr = GR2M()
        inputs = Matrix.from_data('inputs', np.ascontiguousarray(data[:, :2]))
        gr.allocate(inputs, 1)

        # Calibration object
        calib = CalibrationGR2M()

        # Sample parameters
        nsamples = 50
        samples = calib._calparams.clone(nsamples)
        samples.random()

        # loop through parameters
        for i in range(nsamples):

            # Generate obs
            samples.iens = i
            gr.params = calib.cal2true(samples.data)
            expected = gr.params.copy()
            gr.initialise()
            gr.run()

            # Produce theoretical observation
            # with error corruption
            err = np.random.uniform(-1, 1, gr.outputs.shape[0]) * 1e-2
            obs = Matrix.from_data('obs', gr.outputs[:,0].copy()+err)

            # Calibrate
            calib.setup(obs, inputs)

            ini, explo, explo_ofun = calib.explore()
            _, _, ofun = calib.fit(ini, iprint=0)
            final = calib.model.params

            err = np.abs(final-expected)/np.abs(expected)
            ck = np.max(err) < 1e-3

            if i%5 == 0:
                print('\t\tsample {0:02d} : max err = {1:3.3e} ofun = {2}'.format(i,
                            np.max(err), ofun))

            self.assertTrue(ck)

if __name__ == '__main__':
    unittest.main()
