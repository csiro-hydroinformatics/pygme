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
        kn = KNN()
        str_kn = '%s' % kn


    def test_knn_dumb(self):
        nval = 5000
        nvar = 5
        var = np.random.uniform(0, 1, (nval, nvar))
        weights = np.ones(nval)

        kn = KNN(var, weights)

        nrand = 100
        kn.allocate(nrand)
        gr.initialise(54)

        rand = random.uniform(0, 1, nval)
        gr.inputs = rand

        gr.run()

if __name__ == '__main__':
    unittest.main()
