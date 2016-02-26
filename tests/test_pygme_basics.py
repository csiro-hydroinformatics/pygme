import os
import re
import math
import unittest

import time

import numpy as np
import pandas as pd

from calendar import month_abbr as month

from pygme.models.basics import MonthlyPattern, Clip, Node


class MonthlyPatternTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> MonthlyPatternTestCase')

    def test_print(self):
        dm = MonthlyPattern()
        str_dm = '%s' % dm

    def test_monthlypattern1(self):
        dm = MonthlyPattern()

        for i in range(12):
            dm.config[month[i+1]] = 1. + np.sin(float(i)/12 * np.pi) * 10.

        dt = pd.date_range('2010-01-01', '2020-12-31')
        nval = len(dt)

        # Maximum extraction
        inputs = np.zeros((nval, 0))
        dm.allocate(inputs, 1)

        dm.initialise()
        dm.run()
        extrac = pd.Series(dm.outputs[:, 0], index=dt)
        extracm = extrac.resample('MS', 'sum')
        v1 = extracm.groupby(extracm.index.month).mean().values
        v2 = extracm.groupby(extracm.index.month).std().values
        v2[np.isnan(v2)] = 0.

        ck = np.allclose(v1, dm.config.data)
        self.assertTrue(ck)

        ck = np.all(v2 < 1e-6)
        self.assertTrue(ck)


class ClipTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> ClipTestCase')

    def test_print(self):
        dm = Clip()
        str_dm = '%s' % dm

    def test_clip1(self):
        dm = Clip()

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


class NodeTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> NodeTestCase')

    def test_print(self):
        nd = Node(4, 2)
        str_nd = '%s' % nd

    def test_node1(self):
        ninputs = 4
        nd = Node(ninputs)

        nval = 100
        inputs = np.random.uniform(0, 1, (nval, ninputs))
        nd.allocate(inputs)

        nd.run()
        ss = inputs.sum(axis=1)
        ck = np.allclose(ss, nd.outputs[:, 0])

        self.assertTrue(ck)

    def test_node2(self):
        ninputs = 4
        noutputs = 5
        nd = Node(ninputs, noutputs)

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




