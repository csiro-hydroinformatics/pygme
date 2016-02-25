import os
import re
import unittest

from datetime import datetime
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




