import os
import re
import unittest

from itertools import product as prod
from datetime import datetime
import calendar
from dateutil.relativedelta import relativedelta as delta

import numpy as np
import pandas as pd

import c_pygme_models_utils as utils


class UtilsTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> UtilsTestCase')
        FTEST, testfile = os.path.split(__file__)
        self.FOUT = FTEST

    def test_dayinmonth(self):
        ''' Test computation of day in month '''
        dt = pd.date_range('1800-01-01', '2200-12-31', freq='MS')

        for d in dt:
            nb = ((d+delta(months=1)-delta(days=1)) - d).days + 1

            nb2 = utils.daysinmonth(d.year, d.month)

            ck = nb == nb2
            self.assertTrue(ck)

    def test_dayofyear(self):
        ''' Test computation of day of year '''
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
        ''' Test one day shift '''
        dt = pd.date_range('1800-01-01', '2200-12-31', freq='10D')

        for i, d in enumerate(dt):

            d = d - delta(days=1)

            dn = d + delta(days=1)
            dd = np.array([dn.year, dn.month, dn.day])

            dd2 = np.array([d.year, d.month, d.day]).astype(np.int32)
            utils.add1day(dd2)

            ck = np.allclose(dd, dd2)
            self.assertTrue(ck)

    def test_add1month(self):
        ''' Test one month shift '''
        dt = pd.date_range('1800-01-01', '2200-12-31', freq='10D')

        for i, d in enumerate(dt):

            dn = d + delta(months=1)
            dd = np.array([dn.year, dn.month, dn.day])

            dd2 = np.array([d.year, d.month, d.day]).astype(np.int32)
            utils.add1month(dd2)

            ck = np.allclose(dd, dd2)
            self.assertTrue(ck)


    def test_accumulate(self):
        ''' Test accumulation '''
        nval = 365
        a = np.ones(nval)
        start = 20010101.
        mstart = 3
        b = np.zeros((nval, 3))
        utils.accumulate(start, mstart, a, b)

        expected = np.concatenate([np.arange(1, 60), \
                                    np.arange(1, 307)]).astype(float)
        ck = np.allclose(b[:, 0], expected)
        self.assertTrue(ck)

        expected = np.array([2001]*59 + [2002]*306).astype(float)
        ck = np.allclose(b[:, 1], expected)
        self.assertTrue(ck)

        expected = np.concatenate([[np.nan]*59, \
                                np.arange(1, 307)]).astype(float)
        ck = np.allclose(b[59:, 2], expected[59:])
        self.assertTrue(ck)


    def test_rootfind_error(self):
        ''' Test errors generated by the root finding tool
            The function is
            f(x) = -a + x - x / [1+ (x/b)^c ] ^(1/c)
        '''
        # Configure root finder
        niter = np.zeros((1,), np.int32)
        status = np.zeros((1,), np.int32)
        eps = 1e-4
        nitermax = 20

        # Define function
        a = 1.
        b = 10.
        c = 4.
        args = np.array([a, b, c]).astype(np.float64)

        # Error due to initial roots not zero bracketing
        roots = np.array([1., 2.]).astype(np.float64)

        ierr = utils.rootfind_test(1, eps, eps, nitermax, niter,
                status, roots, args)
        self.assertTrue(ierr>0)
        self.assertTrue(status[0] == -1)


    def test_rootfind(self):
        ''' Test root finder solution for
            f(x) = -a + x - x / [1+ (x/b)^c ] ^(1/c)
        '''

        # Configure root finder
        niter = np.zeros((1,), np.int32)
        status = np.zeros((1,), np.int32)
        epsfun = 1e-7
        epsx = 1e-20
        nitermax = 50

        def fun(x, b, c):
            return x-x/(1+(x/b)**c)**(1./c)

        bv = np.linspace(0.1, 100, 10)
        uv = np.linspace(1e-2, 1, 10)
        cv = np.linspace(0.1, 10, 10)

        for u, b, c in prod(uv, bv, cv):
            niter[0] = 0
            status[0] = 0

            # Defines a so that the test function has a root
            # in  [0, +inf[
            a = u*fun(b, b, c)
            roots = np.array([0., 2*b]).astype(np.float64)
            args = np.array([a, b, c]).astype(np.float64)

            ierr = utils.rootfind_test(1, epsx, epsfun, nitermax,
                niter, status, roots, args)

            ck1 = ierr == 0
            ck2 = status[0] > 0 or status[0] == 2
            err = abs(fun(roots[0], b, c)-a)
            ck3 = err < epsfun
            self.assertTrue(ck1 and ck2 and ck3)



if __name__ == "__main__":
    unittest.main()
