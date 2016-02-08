import os
import re
import unittest

from timeit import Timer
import time

import numpy as np

from pygme.models.gr2m import GR2M, CalibrationGR2M
from pygme import calibration
from pygme.model import Matrix

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
        gr.allocate(nval, 9)
        gr.params = [400, 0.9]
        gr.initialise()

        p = np.exp(np.random.normal(0, 2, size=nval))
        pe = np.ones(nval) * 5.
        gr.inputs = np.concatenate([p[:,None], pe[:, None]], axis=1)

        gr.run()


    def test_gr2m_irstea(self):
        fd = '{0}/data/GR2M_timeseries.csv'.format(self.FHERE)
        data = np.loadtxt(fd, delimiter=',', skiprows=1)
        inputs = np.ascontiguousarray(data[:, :2])

        params = [650.7, 0.8]

        # Run
        gr = GR2M()
        gr.allocate(len(inputs), 9)
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
                import pdb; pdb.set_trace()

            self.assertTrue(ck)


    def test_gr2m_irstea_calib(self):
        fd = '{0}/data/GR2M_timeseries.csv'.format(self.FHERE)
        data = np.loadtxt(fd, delimiter=',', skiprows=1)
        inputs = Matrix.fromdata('inputs', np.ascontiguousarray(data[:, :2]))

        calparams_expected = [650.7, 0.8]

        gr = GR2M()
        gr.allocate(inputs.nval, 1)
        gr.inputs = inputs.data

        # Calibration object
        calib = CalibrationGR2M()
        calib.errfun = calibration.ssqe_bias

        # Sample parameters
        nsamples = 50
        samples = calib._calparams.clone(nsamples)
        samples.random()
        samples = samples.transform(calib.cal2true)

        # loop through parameters
        for i in range(nsamples):

            # Generate obs
            samples.iens = i
            gr.params = calib.cal2true(samples.data)
            expected = gr.params.copy()
            gr.initialise()
            gr.run()
            obs = Matrix.fromdata('obs', gr.outputs[:,0].copy())

            # Calibrate
            calib.setup(obs, inputs)
            calib.idx_cal = np.arange(12, inputs.nval)

            ini, explo, explo_ofun = calib.explore()
            final, _, _ = calib.fit(ini, iprint=0)

            err = np.abs(gr.params-expected)
            ck = np.max(err) < 1e-5

            self.assertTrue(ck)

if __name__ == '__main__':
    unittest.main()
