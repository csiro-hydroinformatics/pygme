import sys
import re
import pytest
import itertools

import logging

import time
import math

import numpy as np
import pandas as pd

from scipy.integrate import quad
from scipy.stats import kendalltau, ttest_1samp

from pygme.models.hayami import Hayami, CalibrationHayami, HAYAMI_MAXUH, \
        hayami_kernel, HAYAMI_UHEPS
from pygme.calibration import Calibration, CalibParamsVector, ObjFunSSE, LOGGER

import c_pygme_models_utils
import c_pygme_models_hydromodels

import testdata

np.random.seed(0)

LOGGER.setLevel("INFO")
fmt = "%(asctime)s | %(levelname)s | %(message)s"
date_fmt = "%d %b %H:%M"
ft = logging.Formatter(fmt, date_fmt)
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(ft)
LOGGER.addHandler(sh)

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


@pytest.mark.parametrize("C", [0.01, 0.1, 1., 10.])
@pytest.mark.parametrize("Z", [0.1, 1, 10, 50])
@pytest.mark.parametrize("L", [1e3, 1e4, 1e5])
@pytest.mark.parametrize("timestep", [3600, 86400])
def test_hayami_kernel(C, Z, L, timestep, allclose):
    theta = L / C

    tt = np.linspace(theta / 100, 10 * theta, 100000)
    kernel = hayami_kernel(theta, Z, tt)

    expected = math.sqrt(Z * theta / math.pi)
    expected *= np.exp(Z * (2 - theta / tt - tt / theta))
    expected /= np.sqrt(tt**3)
    assert allclose(kernel[:, 1], expected)

    expected = (kernel[2:, 1] - kernel[:-2, 1]) / 2 / (tt[1] - tt[0])
    err = np.abs(kernel[1:-1, 2] - expected)
    errm = err.max() / np.abs(expected).max()
    assert errm < 5e-3

    expected = (kernel[2:, 2] - kernel[:-2, 2]) / 2 / (tt[1] - tt[0])
    err = np.abs(kernel[1:-1, 3] - expected)
    errm = err.max() / np.abs(expected).max()
    assert errm < 5e-3


@pytest.mark.parametrize("C", [0.01, 0.1, 1., 10.])
@pytest.mark.parametrize("Z", [0.1, 1, 10, 50])
@pytest.mark.parametrize("L", [1e3, 1e4, 1e5])
@pytest.mark.parametrize("timestep", [3600, 86400])
def test_hayami_tbounds(C, Z, L, timestep, allclose):
    theta = L / C

    eps = 1e-5
    bounds = np.zeros(2)
    c_pygme_models_hydromodels.time_bounds_hayami(theta, Z, eps, bounds)
    tlow, thigh = bounds

    t0 = theta * (math.sqrt(16 * Z * Z + 9) - 3) / 4 / Z
    f0 = c_pygme_models_hydromodels.test_hayami_kernel(theta, Z, t0)
    fobj = f0 * eps

    assert tlow < t0
    assert thigh > t0

    fl = c_pygme_models_hydromodels.test_hayami_kernel(theta, Z, tlow)
    el = abs(fl - fobj) / f0
    tol = eps * 1e1
    assert el < tol

    fh = c_pygme_models_hydromodels.test_hayami_kernel(theta, Z, thigh)
    eh = abs(fh - fobj) / f0
    assert eh < tol


@pytest.mark.parametrize("C", [0.01, 0.1, 1., 10.])
@pytest.mark.parametrize("Z", [0.1, 1, 10, 50])
@pytest.mark.parametrize("L", [1e3, 1e4, 1e5])
@pytest.mark.parametrize("iuh", [0, 5, 10, 100])
@pytest.mark.parametrize("timestep", [3600, 86400])
def test_hayami_uh1(C, Z, L, iuh, timestep, allclose):
    theta = L / C

    t1 = float(iuh * timestep)
    tn = float((iuh + 1) * timestep)

    eps = 1e-3
    bounds = np.zeros(2)
    c_pygme_models_hydromodels.time_bounds_hayami(theta, Z, eps, bounds)
    tlow, thigh = bounds
    if  thigh - tlow < timestep / 1000:
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
        u += c_pygme_models_hydromodels.test_integrate_hayami_kernel(a, b, theta, Z)

    u = min(u, 1)

    def fun(x):
        return c_pygme_models_hydromodels.test_hayami_kernel(theta, Z, x)

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
        assert logerr < 5e-2


@pytest.mark.parametrize("C", [0.01, 0.1, 1., 10.])
@pytest.mark.parametrize("Z", [0.1, 1, 10, 50])
@pytest.mark.parametrize("L", [1e3, 1e4, 1e5])
@pytest.mark.parametrize("timestep", [3600, 86400])
def test_hayami_uh2(C, Z, L, timestep, allclose):
    hay = Hayami()
    hay.config.timestep = timestep
    hay.config.length = L
    hay.params.values = [C, Z]

    theta = L / C
    if theta / timestep > HAYAMI_MAXUH / 2:
        pytest.skip("Cannot store full uh")

    ord = hay.ord

    # Check ordinates sum to 1
    assert abs(np.sum(ord) - 1) < HAYAMI_UHEPS * 5.

    # Check enough ordinates
    ipos = np.where(ord > 0)[0]
    assert len(ipos) > 0
    npos = ipos.max() + 1
    nmin = int(theta / timestep)
    assert npos >= nmin


@pytest.mark.parametrize("C", [0.01, 0.1, 1., 10., 50.])
@pytest.mark.parametrize("Z", [0.01, 0.1, 1, 10, 50, 200.])
@pytest.mark.parametrize("L", [1e2, 1e3, 1e4, 1e5, 1e6])
@pytest.mark.parametrize("lateral", [0, 1])
@pytest.mark.parametrize("ntry", range(5))
@pytest.mark.parametrize("timestep", [3600, 86400])
def test_hayami_run_timestep(C, Z, L, lateral, ntry, timestep, allclose):
    nval = 1000
    q1 = np.exp(np.random.normal(0, 2, size=nval))
    inputs = np.ascontiguousarray(q1[:,None])

    hay = Hayami()

    # Set config
    cfg = hay.config
    cfg.timestep = timestep
    cfg.lateral = lateral
    cfg.length = L

    if L / C / timestep > 2 * nval:
        pytest.skip("Very long UH compared to data")

    # Set params
    hay.params.C = C
    hay.params.Z = Z
    uh = hay.ord
    if any(np.isnan(uh)):
        pytest.skip("UH has nan")

    # Allocate
    hay.allocate(inputs, 2)

    t0 = time.time()
    hay.initialise()
    hay.run()
    t1 = time.time()

    # Check model is not drifting
    ord = hay.ord
    ipos = 1 - ord.cumsum() > 1e-3
    nmax = np.where(ipos)[0].max() if ord[0] < 1 - 1e-3 else 1
    if nmax < nval / 3 and nmax > 3:
        vr = hay.outputs[:, 1]
        dvr = np.diff(hay.outputs[:, 1])
        cnt = pd.Series(dvr > 0).value_counts() / nval
        assert all(cnt < 0.9)

    # Check mass balance
    vr = hay.states.values[1]
    si = np.sum(inputs) * cfg.timestep
    so = np.sum(hay.outputs[:,0]) * cfg.timestep
    B = si - so - vr
    atol = 1e-8
    assert abs(B / so) < atol

    # Check outputs
    if hay.ord[0] > 1 - HAYAMI_UHEPS:
        expected = inputs[:, 0]
    elif lateral == 0:
        expected = np.convolve(inputs[:, 0], hay.ord)[:nval]
    else:
        cin = inputs[:, 0].cumsum()
        expected = (cin - np.convolve(cin, hay.ord)[:nval]) * timestep / L * C

    assert allclose(expected, hay.outputs[:, 0], atol=1e-6)

    # Check execution time
    dta = 1000 * (t1 - t0) / nval * 365.25
    assert dta < 4. # Less than 4ms per year sim


@pytest.mark.parametrize("nseries", [1, 2, 3]) #range(1, 16))
@pytest.mark.parametrize("C", [0.1, 1., 10.])
@pytest.mark.parametrize("Z", [0.1, 1., 10.])
@pytest.mark.parametrize("L", [1e2, 1e3, 1e4])
@pytest.mark.parametrize("lateral", [0, 1])
def test_hayami_calibrate(nseries, C, Z, L, lateral):
    hay = Hayami()
    hay.config.length = L
    hay.config.lateral = L

    warmup = 100
    objfun = ObjFunSSE()

    # Get data
    data = testdata.read(f"GR4J_timeseries_{nseries:02d}.csv",
                         source='output', has_dates=False)
    inputs = np.ascontiguousarray(\
                    data.loc[-1000:, ['Qsim']], \
                    np.float64)

    nval = len(inputs)
    if L / C > nval / 2:
        pytest.skip("UH too long")

    # Run hayami first
    hay.allocate(inputs, 1)
    hay.params.C = C
    hay.params.Z = Z
    hay.initialise()
    hay.run()

    # Add error to outputs
    err = np.random.uniform(-1, 1, hay.outputs.shape[0]) * 1e-4
    obs = hay.outputs[:,0].copy() + err

    # Wrapper function for profiling
    calib = CalibrationHayami(objfun=objfun)
    final, ofun, _, _ = calib.workflow(obs, inputs, iprint=20,
                                       maxfun=100000, ftol=1e-8)

    # Test if error on outputs
    warmup = calib.warmup
    sim = calib.model.outputs[:, 0]
    rerr = np.abs(np.arcsinh(obs[warmup:]) - np.arcsinh(sim[warmup:]))
    rerrmax = np.percentile(rerr, 90) # leaving aside 10% of the series
    assert rerrmax < 1e-4
