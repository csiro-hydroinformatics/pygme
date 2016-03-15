import os
import re
import unittest

from datetime import datetime
import calendar
from dateutil.relativedelta import relativedelta as delta

import numpy as np
import pandas as pd

import c_pygme_models_utils as utils


class UtilsTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> UtilsTestCase')
        FTEST, testfile = os.path.split(__file__)
        self.FOUT = FTEST

    def test_dayinmonth(self):
        dt = pd.date_range('1800-01-01', '2200-12-31', freq='MS')

        for d in dt:
            nb = ((d+delta(months=1)-delta(days=1)) - d).days + 1

            nb2 = utils.daysinmonth(d.year, d.month)

            ck = nb == nb2
            self.assertTrue(ck)

    def test_dayofyear(self):
        dt = pd.date_range('1800-01-01', '2200-12-31', freq='10D')

        for d in dt:
            nb = (d-datetime(d.year, 1, 1)).days + 1

            # Correct for leap years
            if calendar.isleap(d.year) and d.month > 2:
                nb -=1

            nb2 = utils.dayofyear(d.month, d.day)

            ck = nb == nb2
            self.assertTrue(ck)


    def test_add1day(self):
        dt = pd.date_range('1800-01-01', '2200-12-31', freq='MS')

        for i, d in enumerate(dt):

            d = d - delta(days=1)

            dn = d + delta(days=1)
            dd = np.array([dn.year, dn.month, dn.day])

            dd2 = np.array([d.year, d.month, d.day]).astype(np.int32)
            utils.add1day(dd2)

            ck = np.allclose(dd, dd2)
            self.assertTrue(ck)

    def test_accumulate(self):

        nval = 365
        a = np.ones(nval)
        start = 20010101.
        mstart = 3
        b = a*0.
        utils.accumulate(start, mstart, a, b)

        expected = np.array(range(1, 60) + range(1, 307)).astype(float)
        ck = np.allclose(b, expected)
        self.assertTrue(ck)


