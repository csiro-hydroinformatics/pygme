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
        params = [0.5, 10., 0.]
        dum = Dummy()
        inputs = np.random.uniform(0, 1, (1000, 2))
        dum.allocate(inputs, 2)

        dum.params = params
        dum.run()
        obs = dum.outputs[:, 0].copy()

        calib = CalibrationDummy(warmup=10)
        calib.setup(Matrix.from_data('obs', obs),
                    Matrix.from_data('inputs', inputs))

        str = '{0}'.format(calib)


    def test_calibration2(self):
        inputs = np.random.uniform(0, 1, (1000, 2))
        obs = np.random.uniform(0, 1, 1000)
        calib = CalibrationDummy(warmup=10)

        try:
            calib.index_cal = obs==obs
        except ValueError as e:
            pass

        self.assertTrue(e.message.startswith('No obsdata'))


    def test_calibration3(self):
        inputs = Matrix.from_data('inputs', np.random.uniform(0, 1, (1000, 2)))
        params = [0.5, 10., 0.]
        dum = Dummy()
        dum.allocate(inputs, 2)
        dum.initialise()

        dum.params = params
        dum.run()
        obs = Matrix.from_data('obs', dum.outputs[:, 0])

        calib = CalibrationDummy(warmup=10)
        calib.setup(obs, inputs)
        calib.index_cal = np.arange(10, obs.nval)

        start, explo, explo_ofun = calib.explore(iprint=0, nsamples=50)

        final, out, _ = calib.fit(start, iprint=0,
                maxfun=100000, ftol=1e-8)

        self.assertTrue(np.allclose(calib.model.params, params))


    def test_calibration4(self):
        inputs = Matrix.from_data('inputs',
                np.random.uniform(0, 1, (1000, 2)))
        params = [0.5, 10., 0.]
        dum = Dummy()
        dum.allocate(inputs, 2)

        dum.params = params
        dum.run()
        obs = Matrix.from_data('obs', dum.outputs[:, 0])

        calib = CalibrationDummy(warmup=10)
        index_cal = np.arange(10, obs.nval)
        calib.run(obs, inputs, index_cal, iprint=0,
                maxfun=100000, ftol=1e-8)

        self.assertTrue(np.allclose(calib.model.params, params))


class CrossValidationTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> CrossValidationTestCase')

    def test_xv1(self):

        dum = Dummy()
        inputs = Matrix.from_data('inputs',
                np.random.uniform(0, 1, (20, 2)))
        dum.allocate(inputs, 2)
        dum.initialise()

        obs = inputs.clone()
        calib = CalibrationDummy(warmup=2)
        calib.setup(obs, inputs)

        xv = CrossValidation(calib=calib)

        # Set leaveout scheme
        xv.set_periods(scheme='leaveout', nperiods=4)
        start_end = [[per['index_start'], per['index_end'],
                        per['index_cal'][0], per['index_cal'][-1]]
                            for per in xv._calperiods]

        # Set split sample test scheme
        xv.set_periods(scheme='split', nperiods=4, warmup=2)
        start_end = [[per['index_start'], per['index_end'],
                        per['index_cal'][0], per['index_cal'][-1]]
                            for per in xv._calperiods]

        expected = [[0, 5, 2, 5], [4 ,9, 6, 9],
                        [8, 13, 10, 13], [12, 17, 14 ,17]]
        self.assertTrue(start_end == expected)


    def test_xv2(self):

        dum = Dummy()
        inputs = Matrix.from_data('inputs',
                np.random.uniform(0, 1, (20, 2)))
        dum.allocate(inputs, 2)
        dum.initialise()

        obs = inputs.clone()

        per = [[0, 5], [4, 9], [8, 13], [12, 17]]
        params = []
        for i in range(4):
            dum.params = [(i+1.)*10, (i+1.)*10., 0.]
            params.append(dum.params.copy())

            idx = range(per[i][0], per[i][1]+1)
            dum.initialise()
            dum.index_start = idx[0]
            dum.index_end = idx[-1]
            dum.run()

            obs.data[idx, :] = dum.outputs[idx, :]

        calib = CalibrationDummy(warmup=10)
        calib.setup(obs, inputs)

        xv = CrossValidation(calib=calib)

        # Run XV scheme
        xv.set_periods('split', 4)
        xv.run(iprint=0, maxfun=100000, ftol=1e-8)

        for i, per in enumerate(xv._calperiods):
            test = per['params']
            expected = params[i]
            self.assertTrue(np.allclose(test, expected))


