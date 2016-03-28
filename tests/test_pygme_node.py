import os
import re
import math
import unittest

import time

import numpy as np
import pandas as pd

from calendar import month_abbr as month

from pygme.models.node import NodeModel
from pygme.data import Matrix
from pygme.calibration import Calibration, ErrorFunctionQuantileReg


class NodeModelTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> NodeModelTestCase')

    def test_print(self):
        nd = NodeModel(4, 2)
        str_nd = '%s' % nd

    def test_nodemodel_sum(self):
        ninputs = 4
        nd = NodeModel(ninputs)

        nval = 100
        inputs = np.random.uniform(0, 1, (nval, ninputs))
        nd.allocate(inputs)

        nd.run()
        ss = inputs.sum(axis=1)
        ck = np.allclose(ss, nd.outputs[:, 0])

        self.assertTrue(ck)

    def test_nodemodel_split(self):
        ninputs = 4
        noutputs = 5
        nd = NodeModel(ninputs, noutputs)

        nval = 100
        inputs = np.random.uniform(0, 1, (nval, ninputs))
        nd.allocate(inputs, noutputs)

        nd.run()
        ss = inputs.sum(axis=1)
        p = np.diag([1./noutputs] * noutputs)
        o = np.dot(np.repeat(ss.reshape((nval, 1)), noutputs, axis=1), p)
        ck = np.allclose(o, nd.outputs)
        self.assertTrue(ck)

        nd.params = range(1, noutputs+1)
        nd.run()
        ss = inputs.sum(axis=1)
        p = np.diag(nd.params/np.sum(nd.params))
        o = np.dot(np.repeat(ss.reshape((nval, 1)), noutputs, axis=1), p)
        ck = np.allclose(o, nd.outputs)
        self.assertTrue(ck)

    def test_nodemodel_clip(self):
        dm = NodeModel()

        dm.config['min'] = 1.
        dm.config['max'] = 5.

        # Maximum extraction
        nval = 1000
        inputs = np.random.uniform(0, 10, (nval, 1))
        dm.allocate(inputs, 1)

        dm.initialise()
        dm.run()

        clip = np.clip(inputs[:,0], dm.config['min'], dm.config['max'])
        ck = np.allclose(dm.outputs[:, 0], clip)

        self.assertTrue(ck)


