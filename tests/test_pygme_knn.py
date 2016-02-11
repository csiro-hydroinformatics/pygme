import os
import re
import unittest

from timeit import Timer
import time

import numpy as np

from pygme.models.knn import KNN
from pygme import calibration
from pygme.model import Matrix

class KNNTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> KNNTestCase')
        filename = os.path.abspath(__file__)
        self.FHERE = os.path.dirname(filename)


    def test_print(self):
        nval = 5000
        nvar = 5
        knn_var = np.random.uniform(0, 1, (nval, nvar))
        weights = np.ones(nval)

        kn = KNN(knn_var, weights)

        str_kn = '%s' % kn


    def test_knn_dumb(self):
        nval = 120
        nvar = 5

        ii = np.arange(nval).astype(float)
        u = np.sin(ii/12-np.pi/2)+1
        knn_var = np.repeat(u[:, None], nvar, axis=1) + np.random.uniform(-0.1, 0.1, (nval, nvar))
        weights = np.ones(nval)

        kn = KNN(knn_var, weights)
        kn.params[2] = 12

        nrand = 100
        kn.allocate(nrand)
        kn.initialise()

        rand = np.random.uniform(0, 1, nrand)
        kn.inputs = rand

        kn.run()
        import pdb; pdb.set_trace()

if __name__ == '__main__':
    unittest.main()
