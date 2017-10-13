import os
import re
import unittest

from timeit import Timer
import time
import math

import numpy as np

from hydrodiy.stat.transform import BoxCox
from pygme.calibration import Calibration
from pygme.calibration import ObjFunSSE, ObjFunBCSSE

from dummy import Dummy, CalibrationDummy

BC = BoxCox()

class ObjFunTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> ErrFunctionTestCase')

        nval = 1000
        obs = np.random.uniform(-0.1, 1, size=nval)
        idx = np.random.choice(np.arange(nval), nval//100)
        obs[idx] = np.nan
        self.obs = obs

        sim = np.random.uniform(-0.1, 1, size=nval)
        idx = np.random.choice(np.arange(nval), nval//100)
        sim[idx] = np.nan
        self.sim = sim


    def test_print(self):
        of = ObjFunBCSSE(0.2)
        print(of)


    def test_SSE(self):
        of = ObjFunSSE()
        value = of.compute(self.obs, self.sim)
        err = self.obs-self.sim
        expected = np.nansum(err*err)
        self.assertTrue(np.allclose(value, expected))


    def test_BCSSE(self):
        for lam in [0.1, 0.5, 1., 2]:
            of = ObjFunBCSSE(lam)
            value = of.compute(self.obs, self.sim)
            BC['lambda'] = lam
            err = BC.forward(self.obs)-BC.forward(self.sim)
            expected = np.nansum(err*err)
            self.assertTrue(np.isclose(value, expected))


class CalibrationTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> CalibrationTestCase')

    def test_calibration1(self):
        params = [0.5, 10., 0.]
        dum = Dummy()
        inputs = np.random.uniform(0, 1, (1000, 2))
        dum.allocate(inputs)

        dum.params.values = params
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


if __name__ == "__main__":
    unittest.main()
