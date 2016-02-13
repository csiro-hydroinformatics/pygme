import os
import re
import unittest

from timeit import Timer
import time

import numpy as np

from pygme import data
from pygme.data import Matrix
from pygme.calibration import Calibration, CrossValidation

from dummy import Dummy, CalibrationDummy

data.set_seed(100)


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
        calib.setup(Matrix.from_data('obs', obs),
                    Matrix.from_data('inputs', inputs))

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
        inputs = Matrix.from_data('inputs', np.random.uniform(0, 1, (1000, 2)))
        params = [0.5, 10., 0.]
        dum = Dummy()
        dum.allocate(inputs.nval, 2)
        dum.initialise()
        dum.inputs = inputs.data

        dum.params = params
        dum.run()
        obs = Matrix.from_data('obs', dum.outputs[:, 0])

        calib = CalibrationDummy()
        calib.setup(obs, inputs)
        calib.idx_cal = np.arange(obs.nval)

        start, explo, explo_ofun = calib.explore(iprint=0, nsamples=50)

        final, out, _ = calib.fit(start, iprint=0,
                maxfun=100000, ftol=1e-8)

        self.assertTrue(np.allclose(calib.model.params, params))


    def test_calibration4(self):
        inputs = Matrix.from_data('inputs', np.random.uniform(0, 1, (1000, 2)))
        params = [0.5, 10., 0.]
        dum = Dummy()
        dum.allocate(inputs.nval, 2)
        dum.inputs = inputs.data

        dum.params = params
        dum.run()
        obs = Matrix.from_data('obs', dum.outputs[:, 0])

        calib = CalibrationDummy()
        idx_cal = np.arange(obs.nval)
        calib.fullfit(obs, inputs, idx_cal, iprint=0,
                maxfun=100000, ftol=1e-8)

        self.assertTrue(np.allclose(calib.model.params, params))


class CrossValidationTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> CrossValidationTestCase')

    def test_xv1(self):

        inputs = Matrix.from_data('inputs', np.random.uniform(0, 1, (20, 2)))
        dum = Dummy()
        dum.allocate(inputs.nval, 2)
        dum.initialise()
        dum.inputs = inputs.data

        obs = inputs.clone()

        per = [[2, 5], [6, 9], [10, 13], [14, 17]]
        for i in range(4):
            dum.params = [(i+1.)/10, i+1., 0.]
            dum.run()

            idx = range(per[i][0], per[i][1]+1)
            obs.data[idx, :] = dum.outputs[idx, :]

        calib = CalibrationDummy()
        calib.setup(obs, inputs)

        xv = CrossValidation(calib=calib)

        # Set leaveout scheme
        xv.set_periods(scheme='leaveout', nperiods=9, warmup=2)
        start_end = [[per['idx_start'], per['idx_end'],
                        per['idx_cal'][0], per['idx_cal'][-1]]
                            for per in xv._calperiods]


        # Set split sample test scheme
        xv.set_periods(scheme='split', nperiods=4, warmup=2)
        start_end = [[per['idx_start'], per['idx_end'],
                        per['idx_cal'][0], per['idx_cal'][-1]]
                            for per in xv._calperiods]

        expected = [[0, 5, 2, 5], [4 ,9, 6, 9],
                        [8, 13, 10, 13], [12, 17, 14 ,17]]
        self.assertTrue(start_end == expected)

        xv.run()

        for i, per in enumerate(xv._calperiods):
            params = per['params']
            expected = [(i+1.)/10, i+1., 0.]
            self.assertTrue(np.allclose(params, expected))

