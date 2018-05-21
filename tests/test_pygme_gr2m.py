import os, sys
import re
import unittest

from timeit import Timer
import time

import numpy as np
from pygme.models.gr2m import GR2M, CalibrationGR2M
from pygme.calibration import ObjFunSSE

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

        gr.X1 = 400
        gr.X2 = 0.9
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
        gr.X1 = params[0]
        gr.X2 = params[1]
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

        gr = GR2M()
        inputs = np.ascontiguousarray(data[:, :2])
        gr.allocate(inputs, 1)

        # Calibration object
        calib = CalibrationGR2M(objfun=ObjFunSSE())

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
            err = np.abs(final-expected)/np.abs(expected)
            ck = np.max(err) < 1e-3

            if i%5 == 0:
                print('\t\tsample {0:02d} : max err = {1:3.3e} ofun = {2}'.format(i,
                            np.max(err), ofun))

            self.assertTrue(ck)


    def test_gr2m_calib_fixed(self):

        fd = '{0}/data/GR2M_timeseries.csv'.format(self.FHERE)
        data = np.loadtxt(fd, delimiter=',', skiprows=1)

        gr = GR2M()
        inputs = np.ascontiguousarray(data[:, :2])
        gr.allocate(inputs, 1)

        # Calibration object
        calib1 = CalibrationGR2M(objfun=ObjFunSSE())
        calib2 = CalibrationGR2M(objfun=ObjFunSSE(), fixed={'X1':200})

        # Generate obs
        expected = [650.7, 0.8]
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

        self.assertTrue(np.allclose(final1, expected, atol=1e-2))
        self.assertTrue(np.allclose(final2[0], 200))


if __name__ == '__main__':
    unittest.main()
