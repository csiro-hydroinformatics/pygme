import os
import re
import pytest
import itertools

import time
import math

import numpy as np
import pandas as pd

from scipy.integrate import quad

import matplotlib.pyplot as plt

from pygme.models.hayami import Hayami, CalibrationHayami, NORDMAXMAX
from pygme.calibration import Calibration, CalibParamsVector, ObjFunSSE


import c_pygme_models_utils
UHEPS = c_pygme_models_utils.uh_getuheps()

import c_pygme_models_hydromodels

import testdata

np.random.seed(0)


def test_print():
    hay = Hayami()
    str_hay = '%s' % hay


def test_run():
    ierr_id = ''
    hay = Hayami()

    inputs = np.zeros((20, 1))
    inputs[1,0] = 100
    hay.allocate(inputs)
    hay.initialise()
    hay.run()


def test_error1():
    hay = Hayami()
    msg = 'model Hayami: Expected noutputs in .1, 2.'
    with pytest.raises(ValueError, match=msg):
        inputs = np.zeros((20, 1))
        hay.allocate(inputs, 30)


def test_error2():
    hay = Hayami()
    msg = 'model Hayami: Expected 1 inputs'
    with pytest.raises(ValueError, match=msg):
        inputs = np.random.uniform(size=(20, 3))
        hay.allocate(inputs, 2)
        hay.initialise()


#@pytest.mark.parametrize("L", [1e3, 1e4, 1e5])
#@pytest.mark.parametrize("C", [0.01, 0.1, 1., 10.])
#@pytest.mark.parametrize("D", [100, 10000, 1000000])
#@pytest.mark.parametrize("iuh", range(10))

#@pytest.mark.parametrize("L", [1e4])
#@pytest.mark.parametrize("C", [0.1])
#@pytest.mark.parametrize("D", [10000])
#@pytest.mark.parametrize("iuh", [0])

@pytest.mark.parametrize("L", [1e3, 1e4, 1e5])
@pytest.mark.parametrize("C", [0.01, 0.1, 1., 10.])
@pytest.mark.parametrize("D", [1, 100, 10000, 1000000])
def test_hayami_kernel(L, C, D, allclose):
    theta = L / C
    z = C * L / 4 / D
    timestep = 86400

    tt = np.linspace(theta / 100, 10 * theta, 100000)
    kernel = [c_pygme_models_hydromodels.test_hayami_kernel(theta, z, t) for t in tt]
    kernel = np.array(kernel)

    expected = math.sqrt(z * theta / math.pi)
    expected *= np.exp(z * (2 - theta / tt - tt / theta))
    expected /= np.sqrt(tt**3)
    assert allclose(kernel, expected)

    dkernel = [c_pygme_models_hydromodels.test_hayami_kernel_diff(theta, z, t) for t in tt]
    dkernel = np.array(dkernel)
    expected = (kernel[2:] - kernel[:-2]) / 2 / (tt[1] - tt[0])

    err = np.abs(dkernel[1:-1] - expected)
    errm = err.max() / np.abs(expected).max()
    assert errm < 5e-3

    d2kernel = [c_pygme_models_hydromodels.test_hayami_kernel_diff2(theta, z, t) for t in tt]
    d2kernel = np.array(d2kernel)
    expected = (dkernel[2:] - dkernel[:-2]) / 2 / (tt[1] - tt[0])

    err = np.abs(d2kernel[1:-1] - expected)
    errm = err.max() / np.abs(expected).max()
    assert errm < 5e-3


@pytest.mark.parametrize("L", [1e3, 1e4, 1e5])
@pytest.mark.parametrize("C", [0.01, 0.1, 1., 10., 100.])
@pytest.mark.parametrize("D", [1, 100, 10000, 1000000])
def test_hayami_tbounds(L, C, D, allclose):
    theta = L / C
    z = C * L / 4 / D
    timestep = 86400

    eps = 1e-5
    bounds = np.zeros(2)
    c_pygme_models_hydromodels.time_bounds_hayami(theta, z, eps, bounds)
    tlow, thigh = bounds

    t0 = theta * (math.sqrt(16 * z * z + 9) - 3) / 4 / z
    f0 = c_pygme_models_hydromodels.test_hayami_kernel(theta, z, t0)
    fobj = f0 * eps

    assert tlow < t0
    assert thigh > t0

    fl = c_pygme_models_hydromodels.test_hayami_kernel(theta, z, tlow)
    el = abs(fl - fobj) / f0
    tol = eps * 1e1
    assert el < tol

    fh = c_pygme_models_hydromodels.test_hayami_kernel(theta, z, thigh)
    eh = abs(fh - fobj) / f0
    assert eh < tol


@pytest.mark.parametrize("L", [1e3, 1e4, 1e5])
@pytest.mark.parametrize("C", [0.01, 0.1, 1., 10.])
@pytest.mark.parametrize("D", [1, 100, 10000, 1000000])
@pytest.mark.parametrize("iuh", range(20))
def test_hayami_uh1(L, C, D, iuh, allclose):
    theta = L / C
    z = C * L / 4 / D
    timestep = 86400

    t1 = float(iuh * timestep)
    tn = float((iuh + 1) * timestep)

    eps = 1e-3
    bounds = np.zeros(2)
    c_pygme_models_hydromodels.time_bounds_hayami(theta, z, eps, bounds)
    tlow, thigh = bounds
    if  thigh - tlow < timestep / 100:
        pytest.skip("Very narrow kernel. too hard for quad")

    if iuh == 0:
        t1 = max(tlow, t1)
        tn = min(thigh, tn)

    n = 5
    dt = (tn - t1) / n
    u = 0
    for i in range(n):
        a = t1 + i * dt
        b = a + dt
        u += c_pygme_models_hydromodels.test_integrate_hayami_kernel(a, b, theta, z)

    u = min(u, 1)

    def fun(x):
        return c_pygme_models_hydromodels.test_hayami_kernel(theta, z, x)

    expected, err, mess = quad(fun,
                               iuh * timestep,
                               (iuh + 1) * timestep,
                               limit=1000,
                               full_output=1)
    if expected < 0 or expected > 1 or err > 1e-3:
        pytest.skip("quad returning negative, >1 or err > 1e-3")

    lu = math.log(max(1e-100, u))
    le = math.log(max(1e-100, expected))
    logerr = abs(lu - le)

    if expected > 1e-3:
        assert logerr < 3e-2

@pytest.mark.parametrize("L", [1e3, 1e4, 1e5])
@pytest.mark.parametrize("C", [0.01, 0.1, 1., 10.])
@pytest.mark.parametrize("D", [1, 100, 10000, 1000000])
def test_hayami_uh2(L, C, D, allclose):
    theta = L / C
    z = C * L / 4 / D
    timestep = 86400

    L = 1e5
    L0 = 1e4
    hay = Hayami()
    hay.config.length = L
    hay.config.length_ref = L0
    hay.config.timestep = timestep

    eta = theta * L0 / L / timestep
    zeta = z * L0 / L
    hay.params.values = [eta, zeta]
    ord = hay.params.uhs[0][1].ord

    # Check enough ordinates
    npos = (1 - ord.cumsum() > 1e-10).sum()
    nmin = int(theta / timestep)
    assert npos >= nmin

    # Check ordinates sum to 1
    assert abs(np.sum(ord) - 1) < UHEPS * 10



def test_max_invv():
    pytest.skip("WIP")

    hay = Hayami()
    # Test if maximum uh length can be set
    length, timestep = 100e3, 3600 # 100km / 1hr
    hay.config.length = length
    hay.config.timestep = timestep

    # Umax = NORDMAXMAX/length*timestep
    hay.alpha = 1.
    # .. creates an error if the max ordinate is not controlled
    hay.U = (NORDMAXMAX+1)/length*timestep


def test_massbalance():
    pytest.skip("WIP")

    nval = 1000
    q1 = np.exp(np.random.normal(0, 2, size=nval))
    inputs = np.ascontiguousarray(q1[:,None])

    hay = Hayami()

    # Set config
    cfg = hay.config
    cfg.timestep = 86400 # daily model
    cfg.length = 86400 # 86.4 km reach
    cfg.flowref = 50 # qstar = 1 m3/s

    # Set outputs
    hay.allocate(inputs, 4)

    print("")
    for theta2 in [1, 2]:
        hay.config.storage_expon = theta2

        # Run
        UU = np.linspace(0.1, 20, 20)
        aa = np.linspace(0., 1., 20)
        dta = 0
        count = 0

        for U, alpha in itertools.product(UU, aa):

            t0 = time.time()

            hay.params.U = U
            hay.params.alpha = alpha
            hay.initialise()
            hay.run()

            t1 = time.time()
            dta += 1000 * (t1-t0) / nval * 365.25

            v0 = 0
            vr = hay.outputs[-1, 2]
            v1 = hay.outputs[-1, 3]
            si = np.sum(inputs) * cfg.timestep
            so = np.sum(hay.outputs[:,0]) * cfg.timestep

            B = si - so - v1 - vr + v0
            ck = abs(B/so) < 1e-10

            assert ck

        dta /= (len(UU) * len(aa))
        print(('\t\ttheta2={0} - Average runtime'+\
                ' = {1:.5f} ms/yr').format( \
                    theta2, dta))


def test_hayami_lag():
    pytest.skip("WIP")
    nval = 1000
    q1 = np.exp(np.random.normal(0, 2, size=nval))
    inputs = np.ascontiguousarray(q1[:,None])

    hay = Hayami()

    # Set configuration
    cfg = hay.config
    cfg.timestep = 86400 # daily model
    cfg.length = 86400 # 86.4 km reach
    cfg.flowref = 50 # qstar = 1 m3/s

    # Set outputs
    hay.allocate(inputs)

    # Run
    for U in range(1, 11):
        hay.params.U = U
        hay.params.alpha = 1
        hay.initialise()
        hay.run()

        err = np.abs(hay.outputs[U:,0] - inputs[:-U, 0])

        ck = np.max(err) < 1e-10
        assert ck


def test_calibrate_against_it():
    ''' Calibrate lag route against a simulation
        with known parameters '''
    pytest.skip("WIP")
    hay = Hayami()
    warmup = 100
    objfun = ObjFunSSE()

    for i in range(16):
        # Get data
        data = testdata.read('GR4J_timeseries_{0:02d}.csv'.format(i+1), \
                                source='output', has_dates=False)
        inputs = np.ascontiguousarray(\
                        data.loc[-1000:, ['Qsim']], \
                        np.float64)

        params = np.random.uniform(0, 1, size=2)
        params[0] *= 5

        # Run hayami first
        hay.allocate(inputs, 1)
        hay.params.values = params
        hay.initialise()
        hay.run()

        # Add error to outputs
        err = np.random.uniform(-1, 1, hay.outputs.shape[0]) * 1e-4
        obs = hay.outputs[:,0].copy()+err

        # Wrapper function for profiling

        calib = CalibrationHayami(objfun=objfun)
        final, ofun, _, _ = calib.workflow(obs, inputs, \
                                    maxfun=100000, ftol=1e-8)

        # Test if error on outputs
        warmup = calib.warmup
        sim = calib.model.outputs[:, 0]
        rerr = np.arcsinh(obs[warmup:]) - np.arcsinh(sim[warmup:])
        rerrmax = np.percentile(rerr, 90) # leaving aside 10% of the series
        assert rerrmax < 2e-3
