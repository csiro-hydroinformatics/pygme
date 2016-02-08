import os
import re
import unittest

from timeit import Timer
import time

import numpy as np

from pygme.model import Model, Matrix
from pygme.calibration import Calibration
from pygme import calibration

from dummy import Dummy, CalibrationDummy


calibration.set_seed(100)


class CalibrationTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> CalibrationTestCase')

    def test_calibration1(self):
        inputs = np.random.uniform(0, 1, (1000, 2))
        params = [0.5, 10., 0.]
        dum = Dummy()
        dum.allocate(inputs.shape[0], 2)
        dum.inputs = inputs

        dum.params = params
        dum.run()
        obs = dum.outputs[:, 0].copy()

        calib = CalibrationDummy()
        calib.setup(Matrix.fromdata('obs', obs),
                    Matrix.fromdata('inputs', inputs))

        str = '{0}'.format(calib)


    def test_calibration2(self):
        inputs = np.random.uniform(0, 1, (1000, 2))
        obs = np.random.uniform(0, 1, 1000)
        calib = CalibrationDummy()

        try:
            calib.idx_cal = obs==obs
        except ValueError as e:
            pass

        self.assertTrue(e.message.startswith('No obsdata'))


    def test_calibration3(self):
        inputs = Matrix.fromdata('in', np.random.uniform(0, 1, (1000, 2)))
        obs = Matrix.fromdata('obs', np.random.uniform(0, 1, 1000))
        calib = CalibrationDummy()
        calib.setup(obs, inputs)

        try:
            start, explo, explo_ofun = calib.explore(iprint=0, nsamples=10)
        except ValueError as e:
            pass
        self.assertTrue(e.message.startswith('No idx_cal data'))


    def test_calibration4(self):
        inputs = Matrix.fromdata('inputs', np.random.uniform(0, 1, (1000, 2)))
        params = [0.5, 10., 0.]
        dum = Dummy()
        dum.allocate(inputs.nval, 2)
        import pdb; pdb.set_trace()
        dum.initialise()
        dum.inputs = inputs.data

        dum.params = params
        dum.run()
        obs = Matrix.fromdata('obs', dum.outputs[:, 0])

        calib = CalibrationDummy()
        calib.setup(obs, inputs)
        calib.idx_cal = np.arange(obs.nval)

        start, explo, explo_ofun = calib.explore(iprint=0, nsamples=50)

        final, out, _ = calib.fit(start, iprint=0,
                maxfun=100000, ftol=1e-8)

        self.assertTrue(np.allclose(calib.model.params, params))


    def test_calibration5(self):
        return
        inputs = Matrix.fromdata('inputs', np.random.uniform(0, 1, (1000, 2)))
        params = [0.5, 10., 0.]
        dum = Dummy()
        dum.allocate(inputs.nval, 2)
        dum.inputs = inputs.data

        dum.params = params
        dum.run()
        obs = Matrix.fromdata('obs', dum.outputs[:, 0])

        calib = CalibrationDummy()
        idx_cal = np.arange(obs.nval)
        calib.fullfit(obs, inputs, idx_cal, iprint=0,
                maxfun=100000, ftol=1e-8)

        self.assertTrue(np.allclose(calib.model.params, params))

