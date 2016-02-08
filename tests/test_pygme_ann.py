import os
import re
import unittest

from timeit import Timer
import itertools
import time

import numpy as np

from pygme.models.ann import ANN, CalibrationANN
from pygme.models.ann import destandardize, standardize, get_standardize_params
from pygme import calibration
from pygme.model import Matrix


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

            params = get_standardize_params(X, c)
            Un = standardize(X, params)
            X2 = destandardize(Un, params)

            ck = np.allclose(X, X2)
            self.assertTrue(ck)


    def test_run1(self):
        n1 = [10, 100, 1000]
        n2 = [1, 2, 10]
        n3 = [1, 2, 10]

        for nval, ninputs, nneurons in itertools.product(n1, n2, n3):

            # Set random parameters
            ann = ANN(ninputs, nneurons)
            idxL1W, idxL1B, idxL2W, idxL2B = ann.params2idx()
            params = np.zeros(ann._params.nval)
            params[idxL1W] = np.random.uniform(0.8, 1.2, len(idxL1W))
            params[idxL1B] = np.random.uniform(-0.2, 0.2, len(idxL1B))
            params[idxL2W] = np.random.uniform(0.8, 1.2, len(idxL2W))
            params[idxL2B] = np.random.uniform(-0.2, 0.2, len(idxL2B))
            ann.params = params

            # Set inputs
            inputs = Matrix.fromdata('inputs', np.random.uniform(-1, 1, (nval, ninputs)))
            ann.allocate(inputs.nval)
            ann.inputs = inputs.data

            # Standardize inputs
            params_std = get_standardize_params(inputs.data)
            ann.inputs_trans_params = params_std
            ann.standardize_inputs()

            # Run model
            ann.run()
            Q1 = ann.outputs[:, 0]

            L1M, L1C, L2M, L2C = ann.params2matrix()
            S2 = np.tanh(np.dot(ann.inputs_trans.data, L1M) + L1C)
            Q2 = (np.dot(S2, L2M) + L2C)[:,0]

            self.assertTrue(np.allclose(Q1, Q2))


    def test_jacobian(self):
        n1 = [10, 100, 1000]
        n2 = [1, 2, 10]
        n3 = [1, 2, 10]

        for nval, ninputs, nneurons in itertools.product(n1, n2, n3):
            # Set Parameters
            ann = ANN(ninputs, nneurons)
            idxL1W, idxL1B, idxL2W, idxL2B = ann.params2idx()
            params = np.zeros(ann._params.nval)
            params[idxL1W] = np.random.uniform(0.8, 1.2, len(idxL1W))
            params[idxL1B] = np.random.uniform(-0.2, 0.2, len(idxL1B))
            params[idxL2W] = np.random.uniform(0.8, 1.2, len(idxL2W))
            params[idxL2B] = np.random.uniform(-0.2, 0.2, len(idxL2B))
            ann.params = params.copy()

            # Set inputs
            inputs = Matrix.fromdata('inputs', np.random.uniform(-1, 1, (nval, ninputs)))
            ann.allocate(inputs.nval)
            ann.inputs = inputs.data

            # Standardize inputs
            params_std = get_standardize_params(inputs.data)
            ann.inputs_trans_params = params_std
            ann.standardize_inputs()

            # Run model
            ann.run()
            Q1 = ann.outputs[:, 0].copy()

            dx = 1e-6
            jac = np.zeros((len(Q1), ann.nparams))
            for k in range(ann.nparams):
                ann.params = params
                ann.params[k] = params[k] + dx
                ann.run()
                Q2 = ann.outputs_trans.data[:, 0].copy()
                jac[:, k] = (Q2-Q1)/dx

            jac2 = ann.jacobian()

            n = jac.shape[1]
            err = np.zeros(n)
            E = np.abs(jac-jac2)/(1+jac) * 100
            for k in range(n):
                err[k] = np.percentile(E, 90)
            ck = np.max(err[idxL1W]) < 1e-4
            ck = ck & (np.max(err[idxL1B]) < 1e-4)
            ck = ck & (np.max(err[idxL2W]) < 1e-4)
            ck = ck & (np.max(err[idxL2B]) < 1e-4)

            self.assertTrue(ck)


    def test_calibrate(self):
        raise ValueError('This ANN test is failing !!!!')
        return

        ninputs = 2
        nneurons = 1
        calib = CalibrationANN(ninputs, nneurons)
        calib.errfun = calibration.sse

        for id in range(1, 21):
            fd = '{0}/data/GR4J_timeseries_{1:02d}.csv'.format(self.FHERE, id)
            d = np.loadtxt(fd, delimiter=',')
            idx = d[:, 3] < 0
            d[idx, 3] = np.nan

            # Compute monthly data
            month = (d[:, 0]*1e-2).astype(int)
            dm = []
            month_u = np.unique(month)
            for m in month_u:
                idx = month == m
                sm = np.sum(d[idx, :], 0)
                sm[0] = m
                dm.append(sm)
            dm = np.concatenate(dm).reshape((len(month_u), 5))

            # Setup a monthly forecast model
            inputs = Matrix.fromdata('inputs', dm[:, [1, 3]])
            obs = Matrix.fromdata('obs', np.append(dm[1:, 3], np.nan))

            # Run ann first
            calib.setup(obs, inputs)
            calib.idx_cal = ~np.isnan(obs.data)

            ann = calib.model
            idxL1W, idxL1B, idxL2W, idxL2B = ann.params2idx()
            params = np.zeros(ann.nparams)
            params[idxL1W] = np.random.uniform(0.8, 1.2, len(idxL1W))
            params[idxL1B] = np.random.uniform(-0.2, 0.2, len(idxL1B))
            params[idxL2W] = np.random.uniform(0.8, 1.2, len(idxL2W))
            params[idxL2B] = np.random.uniform(-0.2, 0.2, len(idxL2W))

            ann.params = params.copy()
            ann.run()

            # Reset calibration to take into account new obs
            obs = Matrix.fromdata('obs', ann.outputs[:,0].copy())
            calib.cst_inputs = 0.01
            calib.cst_outputs = 0.01
            calib.setup(obs, inputs)

            # Calibrate
            ini, _, _ = calib.explore()
            #ini = np.random.uniform(-1, 1, nparams)
            final, _, final_ofun = calib.fit(ini, iprint=5)

            sim = ann.outputs[:, 0]
            erro = np.abs(obs.data - sim)

            err = np.abs(ann.params - params)/np.abs(params) * 100
            ck = np.max(err) < 1e-4

            print('\t\tTEST CALIB %2d : max abs err = %0.5f' % ( \
                    id, np.max(err)))

            self.assertTrue(ck)

if __name__ == '__main__':
    unittest.main()

