import os
import re
import math
import unittest

import time

import numpy as np

from pygme import calibration


class ErrfunTestCases(unittest.TestCase):


    def setUp(self):
        print('\t=> ErrfunTestCase')

        nval = 1000
        self.nval = nval
        self.obs = np.random.uniform(size=nval)
        self.sim = np.random.uniform(size=nval)

    def test_sse(self):
        err1 = calibration.sse(self.obs, self.sim, None)
        err2 = np.sum((self.obs-self.sim)**2)
        ck = np.allclose(err1, err2)
        self.assertTrue(ck)


    def test_ssqe_bias(self):
        err1 = calibration.ssqe_bias(self.obs, self.sim, None)
        err2 = np.sum((np.sqrt(self.obs)-np.sqrt(self.sim))**2)
        err2 *= (1+abs(np.mean(self.obs)-np.mean(self.sim)))
        ck = np.allclose(err1, err2)
        self.assertTrue(ck)


    def test_sls_llikelihood(self):
        sigma = [2.]
        err1 = calibration.sls_llikelihood(self.obs, self.sim, sigma)
        res = self.obs-self.sim
        err2 = np.sum(res*res)/2/sigma[0]**2
        err2 += self.nval * math.log(sigma[0])
        ck = np.allclose(err1, err2)
        self.assertTrue(ck)



