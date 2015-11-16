import os
import re
import unittest

from timeit import Timer
import itertools
import time

import numpy as np

from useme.models.ann import ANN, CalibrationANN
from useme.models.ann import destandardize, standardize
from useme import calibration


class ANNTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> ANNTestCase')
        filename = os.path.abspath(__file__)
        self.FHERE = os.path.dirname(filename)


    def test_print(self):
        ann = ANN(2, 2)
        str_ann = '%s' % ann


    def test_standardize(self):
        nval = 10000
        nvar = 5
        for i in range(10):
            means = np.random.uniform(-10, 10, nvar)
            cov = np.eye(nvar)
            cov.flat[::(nvar+1)] = np.random.uniform(0, 10, nvar)

            X = np.random.multivariate_normal(means, cov, size=nval)
            c = 1e-5
            X = np.exp(X)-c

            Un, mu, su = standardize(X, c)
            X2 = destandardize(Un, mu, su, c)

            ck = np.allclose(X, X2)
            self.assertTrue(ck)


    def test_run1(self):
        n1 = [10, 100, 1000]
        n2 = [1, 2, 50]
        n3 = [1, 2, 10]

        for nval, ninputs, nneurons in itertools.product(n1, n2, n3):

            inputs = np.random.uniform(-1, 1, (nval, ninputs))

            # Parameters
            ann = ANN(ninputs, nneurons)
 
            idxL1W, idxL1B, idxL2W, idxL2B = ann.params2idx()
            params = np.zeros(ann.params.nval)
            params[idxL1W] = np.random.uniform(0.8, 1.2, len(idxL1W))
            params[idxL1B] = np.random.uniform(-0.2, 0.2, len(idxL1B))
            params[idxL2W] = np.random.uniform(0.8, 1.2, len(idxL2W))
            params[idxL2B] = np.random.uniform(-0.2, 0.2, len(idxL2B))
            ann.params.data = params

            ann.params.data = params
            ann.allocate(len(inputs), 2)
            ann.inputs.data = inputs
            ann.run()

            Q1 = ann.outputs.data[:, 0]

            L1M, L1C, L2M, L2C = ann.params2matrix()
            S2 = np.tanh(np.dot(inputs, L1M) + L1C)
            Q2 = (np.dot(S2, L2M) + L2C)[:,0]

            self.assertTrue(np.allclose(Q1, Q2))


    def test_jacobian(self):
        n1 = [10, 100, 1000]
        n2 = [1, 2, 50]
        n3 = [1, 2, 10]

        for nval, ninputs, nneurons in itertools.product(n1, n2, n3):

            inputs = np.random.uniform(-1, 1, (nval, ninputs))

            # Parameters
            ann = ANN(ninputs, nneurons)
            
            idxL1W, idxL1B, idxL2W, idxL2B = ann.params2idx()
            params = np.zeros(ann.params.nval)
            params[idxL1W] = np.random.uniform(0.8, 1.2, len(idxL1W))
            params[idxL1B] = np.random.uniform(-0.2, 0.2, len(idxL1B))
            params[idxL2W] = np.random.uniform(0.8, 1.2, len(idxL2W))
            params[idxL2B] = np.random.uniform(-0.2, 0.2, len(idxL2B))
            ann.params.data = params.copy()

            ann.allocate(len(inputs), 2)
            ann.inputs.data = inputs
            ann.run()
            Q1 = ann.outputs.data[:, 0].copy()

            dx = 1e-6
            jac = np.zeros((len(Q1), ann.params.nval))
            for k in range(ann.params.nval):
                ann.params.data = params
                ann.params.data[k] = params[k] + dx
                ann.run()
                Q2 = ann.outputs.data[:, 0].copy()
                jac[:, k] = (Q2-Q1)/dx

            jac2 = ann.jacobian()

            err = np.mean(np.abs(jac-jac2)/(1+jac) * 100, axis=0)
            ck = np.mean(err[idxL1W]) < 1e-3
            ck = ck & (np.mean(err[idxL1B]) < 1e-4)
            ck = ck & (np.mean(err[idxL2W]) < 2)
            ck = ck & (np.mean(err[idxL2B]) < 1e-7)
           
            self.assertTrue(ck)


    def test_calibrate(self):
        ninputs = 2
        nneurons = 1
        calib = CalibrationANN(ninputs, nneurons)
        calib.errfun = calibration.sse

        for id in range(1, 21):
            fd = '{0}/data/GR4J_timeseries_{1:02d}.csv'.format(self.FHERE, id)
            d = np.loadtxt(fd, delimiter=',')
            idx = d[:, 3] < 0
            d[idx, 3] = np.nan

            month = (d[:, 0]*1e-2).astype(int)
            dm = []
            month_u = np.unique(month)
            for m in month_u:
                idx = month == m
                sm = np.sum(d[idx, :], 0)
                sm[0] = m
                dm.append(sm)
            dm = np.concatenate(dm).reshape((len(month_u), 5))

            inputs = dm[:, [1, 3]]
            obs = np.append(dm[1:, 3], np.nan)

            # Standardize
            cst = 0.01
            inputs_s, m_I, s_I = standardize(inputs, cst)
            obs_s, m_O, s_O = standardize(obs, cst)

            # Run gr first
            calib.setup(obs_s, inputs_s)
            calib.idx_cal = np.arange(len(obs_s))

            ann = calib.model

            idxL1W, idxL1B, idxL2W, idxL2B = ann.params2idx()
            params = np.zeros(ann.params.nval)
            params[idxL1W] = np.random.uniform(0.8, 1.2, len(idxL1W))
            params[idxL1B] = np.random.uniform(-0.2, 0.2, len(idxL1B))
            params[idxL2W] = np.random.uniform(0.8, 1.2, len(idxL2W))

            ann.run()

            obs = ann.outputs.data[:,0].copy()
            calib.observations.data = obs

            # Calibrate
            ini, _, _ = calib.explore()
            final, _, final_ofun = calib.fit(ini)

            sim = ann.outputs.data[:, 0]
            erro = np.abs(obs - sim)

            err = np.abs(ann.params.data - params)/params * 100
            ck = np.max(err) < 1e-4

            print('\t\tTEST CALIB %2d : max abs err = %0.5f' % ( \
                    id, np.max(err)))

            self.assertTrue(ck)

if __name__ == '__main__':
    unittest.main()

