import os
import re
import math
import unittest

import time

import numpy as np
import pandas as pd

from calendar import month_abbr as month

from pygme.models.demand import Demand


class DemandTestCases(unittest.TestCase):


    def setUp(self):
        print('\t=> DemandTestCase')

    def test_print(self):
        dm = Demand()
        str_dm = '%s' % dm


    def test_demand1(self):
        dm = Demand()

        for i in range(12):
            dm.config[month[i+1]] = 1. + np.sin(float(i)/12 * np.pi) * 10.

        dt = pd.date_range('2010-01-01', '2020-12-31')
        nval = len(dt)

        # Maximum extraction
        inputs = np.zeros((nval, 2))
        inputs[:, 0] = inputs[:, 0] + 100
        dm.allocate(inputs, 3)

        def get_monthly(dm):
            dm.initialise()
            dm.run()
            extrac = pd.Series(dm.outputs[:, 0], index=dt)
            extracm = extrac.resample('MS', 'sum')
            return extracm.groupby(extracm.index.month).mean().values, extracm

        v1, vm = get_monthly(dm)
        ck = np.allclose(v1, dm.config.data)
        self.assertTrue(ck)

        # Check outflow + extraction = inflow
        v2 = np.sum(dm.outputs[:,:2], 1)
        ck = np.allclose(v2, inputs[:, 0])
        self.assertTrue(ck)

        # No extraction
        inputs = np.zeros((nval, 2))
        inputs[:, 0] = 1.
        inputs[:, 1] = 1.
        dm.allocate(inputs, 3)
        v1, _ = get_monthly(dm)
        ck = np.allclose(v1, 0.)
        self.assertTrue(ck)


