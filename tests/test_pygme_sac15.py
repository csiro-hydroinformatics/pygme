import os
import re
import unittest
from pathlib import Path

import time
import math

import numpy as np

from pygme import calibration
from pygme.models.sac15 import SAC15, CalibrationSAC15

import c_pygme_models_utils
UHEPS = c_pygme_models_utils.uh_getuheps()


class SAC15TestCases(unittest.TestCase):


    def setUp(self):
        print('\t=> SAC15TestCase')
        filename = Path(__file__).resolve()
        self.FHERE = filename.parent

    def test_print(self):
        sa = SAC15()
        str_sa = '%s' % sa


    def test_error1(self):
        sa = SAC15()
        try:
            sa.allocate(np.random.uniform(0, 1, (200, 2)), 30)
        except ValueError as  e:
            pass
        self.assertTrue(str(e).startswith('With model sac15, Number of outputs'))


    def test_error2(self):
        sa = SAC15()
        inputs = np.random.uniform(size=(20, 3))
        try:
            sa.allocate(inputs, 5)
            sa.initialise()
        except ValueError as  e:
            pass
        self.assertTrue(str(e).startswith('With model sac15, Number of inputs'))


    def test_uh(self):
        sa = SAC15()
        sa.allocate(np.zeros((10, 2)))

        for Lag in np.linspace(0, 50, 100):
            sa.params.reset()
            sa.params.Lag = Lag

            ordu = sa.params.uhs[0][1].ord
            ck = abs(np.sum(ordu)-1) < UHEPS * 1
            self.assertTrue(ck)


    def test_run1(self):
        sa = SAC15()
        nval = 1000

        p = np.exp(np.random.normal(0, 2, size=nval))
        pe = np.ones(nval) * 5.

        inputs = np.array([p, pe]).T
        sa.allocate(inputs, 6)
        sa.initialise()
        sa.params.reset()
        sa.run()

        out = sa.outputs
        ck = out.shape == (nval, 6)
        self.assertTrue(ck)


    def test_run2(self):
        return
        warmup = 365 * 5
        sa = SAC15()

        fp = self.FHERE / "sac15" / "SAC15_params.csv"
        params = np.loadtxt(fp, delimiter=',')

        for i in range(params.shape[0]):
            fts = self.FHERE / "sac15" / f"SAC15_timeseries_{i+1:02d}.csv"
            data = np.loadtxt(fts, delimiter=',')
            inputs = np.ascontiguousarray(data[:, [1, 2]], np.float64)

            # Run sac15 [block]
            sa.allocate(inputs)
            t0 = time.time()
            sa.params = params[i, [2, 0, 1, 3]]
            sa.initialise()
            sa.run_as_block = True
            sa.run()
            qsim1 = sa.outputs[:,0].copy()
            t1 = time.time()
            dta1 = 1000 * (t1-t0)
            dta1 /= len(qsim1)/365.25

            # Run sac15 [timestep]
            sa._outputs.reset()
            t0 = time.time()
            sa.initialise()
            sa.run_as_block = False
            sa.run()
            qsim2 = sa.outputs[:,0].copy()
            t1 = time.time()
            dta2 = 1000 * (t1-t0)
            dta2 /= len(qsim2)/365.25

            # Compare
            idx = np.arange(inputs.shape[0]) > warmup
            expected = data[idx, 4]

            err = np.abs(qsim1[idx] - expected)
            err_thresh = 5e-2
            ck1 = np.max(err) < err_thresh

            err = np.abs(qsim2[idx] - qsim1[idx])
            err_thresh = 5e-2
            ck2 = np.max(err) < err_thresh

            if not (ck1 and ck2):
                print(('\t\tTEST %2d : max abs err = '
                    '%0.5f < %0.5f ? %s ~ %0.5fms/yr\n') % (i+1, \
                    np.max(err), err_thresh, ck1, ck2))
            else:
                print(('\t\tTEST %2d : max abs err = %0.5f\n\t\t\truntime :' +
                        ' %0.5fms/yr [block] / %0.5fms/yr [ts]\n') % ( \
                    i+1, np.max(err), dta1, dta2))

            self.assertTrue(ck1)
            self.assertTrue(ck2)


    def test_calibrate(self):
        return
        sa = SAC15()
        warmup = 365*6

        calib = CalibrationSAC15()


        fp = self.FHERE / "sac15" / "SAC15_params.csv"
        params = np.loadtxt(fp, delimiter=',')

        for i in range(params.shape[0]):

            fts = self.FHERE / "data" / f"SAC15_timeseries_{i+1:02d}.csv"
            data = np.loadtxt(fts, delimiter=',')
            inputs = Matrix.from_data('inputs',
                    np.ascontiguousarray(data[:, [1, 2]], np.float64))

            # Run sa first
            params_expected = params[i, [2, 0, 1, 3]]
            sa = calib.model
            sa.allocate(inputs, 1)
            sa.params = params_expected
            sa.initialise()
            sa.run()
            obs = Matrix.from_data('obs', sa.outputs[:,0].copy())

            # Calibrate
            calib.setup(obs, inputs)

            start, explo, explo_ofun = calib.explore()
            ieval1 = calib.ieval

            final, _, _ = calib.fit(start, ftol=1e-8)
            ieval2 = calib.ieval - ieval1

            err = np.abs(calib.model.params - params_expected)
            ck = np.max(err) < 1e-7

            print(('\t\tTEST CALIB {0:02d} : max abs err = {1:3.3e}' +
                    ' neval= {2} + {3}').format( \
                        i+1, np.max(err), ieval1, ieval2))

            self.assertTrue(ck)

