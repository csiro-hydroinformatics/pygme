import os
import re
import pytest
from itertools import product as prod

from timeit import Timer
import time

import numpy as np

from pygme.calibration import ObjFunSSE

from pygme.models.turcmezentsev import TurcMezentsev, \
                                    turcm_forward, \
                                    turcm_gradient, \
                                    turcm_backward, \
                                    CalibrationTurcMezentsev

def test_print():
    ''' Test printing function '''
    tm = TurcMezentsev()
    str_tm = '%s' % tm


def test_run(allclose):
    ''' Test model run '''
    tm = TurcMezentsev()

    P = np.linspace(700, 1200, 20)
    PE = np.linspace(1000, 2000, 20)
    inputs = np.concatenate([P[:,None], PE[:, None]], axis=1)
    tm.allocate(inputs, 2)
    tm.run()
    Q1 = tm.outputs[:, 0]
    n = tm.params.values[0]
    Q2 = P*(1-1/(1+(P/PE)**n)**(1/n))
    assert allclose(Q1, Q2)


def test_backward():
    ''' Test backward function '''

    aas = np.linspace(0.05, 2., 20)
    ns = np.linspace(1.1, 4., 20)
    P = 100
    for a, nini in prod(aas, ns):
        E = P/a
        Q = turcm_forward(P, E, nini)
        r = Q/P
        nback, niter = turcm_backward(Q, P, E)
        assert np.isclose(nini, nback, rtol=0., atol=5e-3)


def test_gradient():
    ''' Test backward function '''

    aas = np.linspace(0.05, 2., 20)
    ns = np.linspace(1.1, 4., 20)
    P = 100
    for a, n in prod(aas, ns):
        E = P/a
        r0 = turcm_forward(P, E, n)/P
        dx = 1e-6
        r1 = turcm_forward(P+E*dx, E, n)/(P+E*dx)
        G = turcm_gradient(P, E, n)/P
        assert np.isclose(G, (r1-r0)/dx, rtol=0., atol=1e-5)


def test_calibrate(allclose):
    ''' Test the calibration process '''

    Q = [85.5, \
            331.7, \
            87.5, \
            109.6, \
            60.3, \
            122.4, \
            87.9, \
            57.8, \
            290.2, \
            304.4, \
            24.8, \
            503.4, \
            261.4, \
            626.4, \
            206.4, \
            165.6, \
            348.5, \
            329.5, \
            214.9, \
            297.4, \
            507.2, \
            164.4]

    P = [749.2, \
            1142.0, \
            896.3, \
            820.8, \
            735.7, \
            901.1, \
            800.6, \
            808.5, \
            956.9, \
            954.2, \
            668.7, \
            1152.4, \
            1049.3, \
            1324.2, \
            884.7, \
            783.9, \
            927.0, \
            1034.7, \
            1124.2, \
            841.5, \
            922.4, \
            1020.1]

    E = [1173.7, \
            1110.8, \
            1150.5, \
            1120.3, \
            1141.2, \
            1362.1, \
            1237.0, \
            1217.7, \
            1172.7, \
            1253.1, \
            1421.2, \
            1246.0, \
            1333.7, \
            1116.4, \
            1167.3, \
            1135.9, \
            1177.4, \
            1171.0, \
            1231.1, \
            1068.3, \
            1006.3, \
            1171.0]

    inputs = np.array([P, E]).T
    obs = np.array(Q)
    objfun = ObjFunSSE()
    calib = CalibrationTurcMezentsev(objfun=objfun)
    calib.workflow(obs, inputs)

    tm = calib.model
    assert allclose(tm.params['n'], 1.7190228, atol=1e-3)


