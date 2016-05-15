import os
import re
import unittest

from timeit import Timer
import time
import math

import numpy as np

from pygme import data
from pygme.data import Matrix
from pygme.calibration import Calibration, powertrans
from pygme.calibration import ErrorFunctionSseBias, ErrorFunctionQuantileReg, ErrorFunctionSls

from dummy import Dummy, CalibrationDummy

data.set_seed(100)



class ErrFunctionTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> ErrFunctionTestCase')
        self.obs = np.random.uniform(-1, 1, size=1000)
        self.sim = np.random.uniform(-1, 1, size=1000)

    def test_ssebias(self):
        of = ErrorFunctionSseBias()
        value = of.run(self.obs, self.sim)
        expected = np.mean((self.obs-self.sim)**2)
        self.assertTrue(np.allclose(value, expected))

        of.constants = [1., 1., 0.]
        value = of.run(self.obs, self.sim)
        expected = np.mean(np.abs(self.obs-self.sim))
        self.assertTrue(np.allclose(value, expected))

        of.constants = [0.5, 1., 0.]
        value = of.run(self.obs, self.sim)
        expected = np.nanmean(np.abs(powertrans(self.obs, 0.5)-powertrans(self.sim, 0.5)))
        self.assertTrue(np.allclose(value, expected))

        of.constants = [0.5, 1., 0.1]
        value = of.run(self.obs, self.sim)
        expected = np.nanmean(np.abs(powertrans(self.obs, 0.5)-powertrans(self.sim, 0.5)))
        bias = np.mean(self.obs-self.sim)/(1+abs(np.mean(self.obs)))
        expected *= 1+0.1*bias*bias
        self.assertTrue(np.allclose(value, expected))


    def test_slslikelihood(self):
        sls = ErrorFunctionSls()

        sigma = 5.
        sls.errparams = np.log(sigma)
        value = sls.run(self.obs, self.sim)

        err = self.obs-self.sim
        nval = len(self.obs)
        expected = np.nansum(err*err)/(2*sigma*sigma) + nval * math.log(sigma)
        self.assertTrue(np.allclose(value, expected))




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
        return
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



def get_startend(xv, is_cal=True, is_leave=True):

    label1= ['val', 'cal'][int(is_cal)]
    label2= ['', 'leaveout'][int(is_leave)]

    return [per[re.sub('_$', '', 'ipos_{0}_{1}'.format(label1, label2))]
                    for per in xv._calperiods]


