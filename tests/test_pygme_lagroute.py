import os
import re
import unittest
import itertools

import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from pygme.models.lagroute import LagRoute, CalibrationLagRoute
from pygme.calibration import Calibration, CalibParamsVector, ObjFunBCSSE


import c_pygme_models_utils
UHEPS = c_pygme_models_utils.uh_getuheps()


class LagRouteTestCases(unittest.TestCase):


    def setUp(self):
        print('\t=> LagRouteTestCase')
        FOUT = os.path.dirname(os.path.abspath(__file__))
        self.FOUT = FOUT


    def test_print(self):
        lr = LagRoute()
        str_lr = '%s' % lr


    def test_run(self):

        ierr_id = ''
        lr = LagRoute()

        inputs = np.zeros((20, 1))
        inputs[1,0] = 100
        lr.allocate(inputs, 2)
        lr.initialise()
        lr.run()


    def test_error1(self):
        lr = LagRoute()
        try:
            inputs = np.zeros((20, 1))
            lr.allocate(inputs, 30)
        except ValueError as  err:
            self.assertTrue(str(err).startswith('model LagRoute: Expected noutputs in [1, 4]'))
        else:
            raise ValueError('Problem in error handling')


    def test_error2(self):
        lr = LagRoute()
        try:
            inputs = np.random.uniform(size=(20, 3))
            lr.allocate(inputs, 2)
            lr.initialise()
        except ValueError as  err:
            self.assertTrue(str(err).startswith('model LagRoute: Expected 1 inputs'))
        else:
            raise ValueError('Problem in error handling')


    def test_uh1(self):
        lr = LagRoute()
        for u, a in itertools.product(np.linspace(0, 10, 20), \
                np.linspace(0, 1, 20)):
            lr.params.values = [u, a]
            ord = lr.params.uhs[0][1].ord
            ck = abs(np.sum(ord)-1) < UHEPS
            self.assertTrue(ck)


    def test_uh2(self):
        lr = LagRoute()

        # Set config
        cfg = lr.config
        cfg.timestep = 86400 # daily model
        cfg.length = 86400 # 86.4 km reach
        cfg.flowref = 1 # qstar = 1 m3/s
        cfg.storage_expon = 1 # linear model

        alpha = 1.
        Umin = lr.params.mins[0]
        Umax = lr.params.maxs[0]

        # Set uh
        for U in np.linspace(Umin, Umax, 100):

            lr.params.U = U
            lr.params.alpha = alpha

            ord = lr.params.uhs[0][1].ord
            ck = abs(np.sum(ord)-1) < 1e-5
            self.assertTrue(ck)

            tau = alpha * cfg.length * U
            k = int(tau/cfg.timestep)
            w = tau/cfg.timestep - k
            ck = abs(ord[k]-1+w) < 1e-5
            self.assertTrue(ck)


    def test_massbalance(self):

        nval = 1000
        q1 = np.exp(np.random.normal(0, 2, size=nval))
        inputs = np.ascontiguousarray(q1[:,None])

        lr = LagRoute()

        # Set config
        cfg = lr.config
        cfg.timestep = 86400 # daily model
        cfg.length = 86400 # 86.4 km reach
        cfg.flowref = 50 # qstar = 1 m3/s

        # Set outputs
        lr.allocate(inputs, 4)

        for theta2 in [1, 2]:

            lr.config.storage_expon = theta2

            # Run
            UU = np.linspace(0.1, 20, 20)
            aa = np.linspace(0., 1., 20)
            dta = 0
            count = 0

            for U, alpha in itertools.product(UU, aa):

                t0 = time.time()

                lr.params.U = U
                lr.params.alpha = alpha
                lr.initialise()
                lr.run()

                t1 = time.time()
                dta += 1000 * (t1-t0) / nval * 365.25

                v0 = 0
                vr = lr.outputs[-1, 2]
                v1 = lr.outputs[-1, 3]
                si = np.sum(inputs) * cfg.timestep
                so = np.sum(lr.outputs[:,0]) * cfg.timestep

                B = si - so - v1 - vr + v0
                ck = abs(B/so) < 1e-10

                self.assertTrue(ck)

            dta /= (len(UU) * len(aa))
            print('\t\ttheta2={0} - Average runtime = {1:.5f} ms/yr'.format( \
                theta2, dta))

    def test_lagroute_lag(self):
        nval = 1000
        q1 = np.exp(np.random.normal(0, 2, size=nval))
        inputs = np.ascontiguousarray(q1[:,None])

        lr = LagRoute()

        # Set configuration
        cfg = lr.config
        cfg.timestep = 86400 # daily model
        cfg.length = 86400 # 86.4 km reach
        cfg.flowref = 50 # qstar = 1 m3/s

        # Set outputs
        lr.allocate(inputs)

        # Run
        for U in range(1, 11):
            lr.params.U = U
            lr.params.alpha = 1
            lr.initialise()
            lr.run()

            err = np.abs(lr.outputs[U:,0] - inputs[:-U, 0])

            ck = np.max(err) < 1e-10
            self.assertTrue(ck)


