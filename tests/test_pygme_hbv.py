import os
import re
import unittest

import time

import numpy as np

from pygme.calibration import ObjFunSSE
from pygme.models.hbv import HBV, CalibrationHBV

import c_pygme_models_utils
UHEPS = c_pygme_models_utils.uh_getuheps()


class HBVTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> HBVTestCase')
        filename = os.path.abspath(__file__)
        self.ftest = os.path.dirname(filename)


    def test_print(self):
        hb = HBV()
        str_hb = '%s' % hb


    def test_error1(self):
        hb = HBV()
        try:
            hb.allocate(np.random.uniform(0, 1, (200, 2)), 30)
        except ValueError as  err:
            self.assertTrue(str(err).startswith(\
                                'model HBV: Expected noutputs'))
        else:
            raise ValueError('Problem with error handling')


    def test_error2(self):
        hb = HBV()
        inputs = np.random.uniform(size=(20, 3))
        try:
            hb.allocate(inputs, 5)
            hb.initialise()
        except ValueError as  err:
            self.assertTrue(str(err).startswith(\
                        'model HBV: Expected 2 inputs'))
        else:
            raise ValueError('Problem with error handling')


    def test_run_dimensions(self):
        ''' Allocate HBV '''
        hb = HBV()
        nval = 1000

        p = np.exp(np.random.normal(0, 2, size=nval))
        pe = np.ones(nval) * 5.

        inputs = np.array([p, pe]).T
        hb.allocate(inputs, 10)
        hb.initialise()
        hb.run()

        out = hb.outputs

        ck = out.shape == (nval, 10)
        self.assertTrue(ck)


    def test_run_against_data(self):
        ''' Compare HBV simulation with test data '''
        warmup = 365 * 10
        warmup = 0

        hb = HBV()

        #for i in range(13, 14):
        for i in range(20):
            # Get inputs
            fts = os.path.join(self.ftest, 'output_data', \
                    'HBV_timeseries_{0:02d}.csv'.format(i+1))
            data = np.loadtxt(fts, delimiter=',', skiprows=1)

            nval = 10
            data = data[:nval]
            #inputs = np.ascontiguousarray(data[:, [0, 1]], np.float64)

            # Lag by one time step ?????
            inputs = np.ascontiguousarray(data[1:, [0, 1]], np.float64)
            inputs = np.concatenate([inputs, inputs[-1:]], axis=0)

            # Get parameters
            fp = os.path.join(self.ftest, 'output_data', \
                    'HBV_params_{0:02d}.csv'.format(i+1))
            params = np.genfromtxt(fp, dtype=[('parname', str), \
                                    ('parvalue', np.float64)], \
                                    skip_header=1, delimiter=',')
            params = np.array([p[1] for p in params])

            # Allocate model and set parameters
            hb.allocate(inputs, 13)
            hb.params.values = params

            # .. initiase to same values than TUWmodel run ...
            s0 = data[0, [9, 13, 14]]
            s1 = data[1, [9, 13, 14]]
            sini = 2*s0-s1
            #sini = [50, 2.5, 2.5]
            hb.initialise(states=sini)

            # Run model
            hb.run()

            # Get outputs
            onames = ['Q', 'Q0', 'Q1', 'Q2', 'ETA']
            allnames = np.array(hb.outputs_names)
            cc = [np.where(n == allnames)[0][0] for n in onames]
            sim = hb.outputs[:, cc].copy()

            snames = allnames[-3:]
            states = hb.outputs[:, -3:].copy()

            # Compare with outputs generated from R code
            # Comparison is done after warmup
            idx = np.arange(inputs.shape[0]) > warmup
            expected = data[:, [3, 6, 7, 8, 12]]
            expected_states = data[:, [9, 13, 14]]

            err = np.abs(sim[idx, :] - expected[idx, :])

            # Sensitivity to initial conditionos
            s1 = [0]*3
            Pm = np.mean(inputs[:, 0])
            s2 = [hb.params.FC, hb.params.LSUZ, 2*Pm*hb.params.K2]
            warmup_ideal, sim0, sim1 = hb.inisens(s1, s2, ignore_error=True)

            # Special criteria
            # 5 values with difference hbeater than 1e-5
            # max diff lower than 5e-4
            def fun(x):
                return np.sum(x > 1e-5), np.max(x)

            cka = np.array([fun(err[:, k]) for k in range(err.shape[1])])
            ck = np.all((cka[:, 0] < 5) & (cka[:, 1] < 1e-4))

            print(('\t\tTEST SIM {0:2d} : crit={1} err={2:3.3e}'+\
                    ' warmup={3}').format(i+1, ck, np.max(err), \
                            warmup_ideal))

            try:
                self.assertTrue(ck)
            except:
                import matplotlib.pyplot as plt
                from hydrodiy.plot import putils
                plt.close('all')
                fig, axs = plt.subplots(nrows=3)
                for i in range(3):
                    x1 = expected[:, i]
                    x2 = sim[:, i]
                    title = onames[i]

                    x1 = expected_states[:, i]
                    x2 = states[:, i]
                    title = snames[i]

                    ax = axs.flat[i]
                    ax.plot(x1, label='TUWmodel')
                    ax.plot(x2, label='pygme')
                    tax = ax.twinx()
                    tax.plot(x2-x1, 'k-', lw=0.7)
                    ax.set_title(title)

                    ax.legend()
                fig.tight_layout()
                plt.show()
                import pdb; pdb.set_trace()



    def test_calibrate_against_itself(self):
        ''' Calibrate HBV against a simulation with known parameters '''
        return
        hb = HBV()
        warmup = 365*6
        calib = CalibrationHBV(objfun=ObjFunSSE())

        for i in range(8, 9):
            # Get inputs
            fts = os.path.join(self.ftest, 'output_data', \
                    'HBV_timeseries_{0:02d}.csv'.format(i+1))
            data = np.loadtxt(fts, delimiter=',', skiprows=1)
            inputs = np.ascontiguousarray(data[:, [1, 0]], np.float64)

            # Get parameters
            fp = os.path.join(self.ftest, 'output_data', \
                    'HBV_params_{0:02d}.csv'.format(i+1))
            params = np.loadtxt(fp, delimiter=',', skiprows=1)

            # Run hb first
            hb = calib.model
            hb.allocate(inputs, 1)
            hb.params.values = params
            hb.initialise()
            hb.run()

            # Calibrate
            err = np.random.uniform(-1, 1, hb.outputs.shape[0]) * 1e-4
            sim0 = hb.outputs[:, 0].copy()
            obs = sim0+err

            t0 = time.time()
            final, ofun, _ = calib.workflow(obs, inputs, \
                                        maxfun=100000, ftol=1e-8)
            t1 = time.time()
            dt = (t1-t0)/len(obs)*365.25

            # Test error on parameters
            err = np.abs(final-params)
            ck1 = np.max(err) < 1e-2

            idx = np.arange(len(sim0)) > warmup
            errs = np.abs(hb.outputs[idx, 0]-sim0[idx])
            ck2 = np.max(errs) < 1e-3

            print(('\t\tTEST CALIB {0:02d} : max abs err = {1:3.3e}'+\
                    ' dt={2:3.3e} sec/yr').format(\
                        i+1, np.max(err), dt))

            if not (ck1 or ck2):
                import pdb; pdb.set_trace()

            self.assertTrue(ck)


