import os, sys, re
import pytest
from pathlib import Path
import logging

import time
import math

import numpy as np
import pandas as pd
from scipy.optimize import fmin_bfgs

from hydrodiy.stat import sutils

from pygme.calibration import ObjFunSSE, LOGGER
from pygme.models.hbv import HBV, CalibrationHBV
from pygme.models.hbv import hbv_trans2true, hbv_true2trans
from pygme.models.hbv import HBV_TMEAN, HBV_TCOV

import c_pygme_models_utils
UHEPS = c_pygme_models_utils.uh_getuheps()

filename = Path(__file__).resolve()
FHERE = filename.parent

LOGGER.setLevel(logging.INFO)
fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
ft = logging.Formatter(fmt)
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(ft)
LOGGER.addHandler(sh)


def get_hbv_data(catchment):
    fp = FHERE / "hbv" / f"HBV_params_{catchment:02d}.csv"
    params = pd.read_csv(fp, index_col="parname").squeeze()
    params.index = params.index.str.upper()

    fts = FHERE / "hbv" / f"HBV_timeseries_{catchment:02d}.csv"
    data = pd.read_csv(fts)
    inputs = np.ascontiguousarray(data.iloc[:, [0, 1]], np.float64)
    return data, inputs, params


def test_print():
    sa = HBV()
    LOGGER.info(str(sa))


def test_error1():
    sa = HBV()
    msg = "model HBV: Expected noutputs in"
    with pytest.raises(ValueError, match=msg):
        sa.allocate(np.random.uniform(0, 1, (200, 2)), 30)


def test_error2():
    sa = HBV()
    inputs = np.random.uniform(size=(20, 3))
    msg = "model HBV: Expected 2 inputs, got"
    with pytest.raises(ValueError, match=msg):
        sa.allocate(inputs, 5)
        sa.initialise()


def test_params_transform(allclose):
    nparamslib = 10000
    model = HBV()
    tplib = sutils.lhs_norm(nparamslib, HBV_TMEAN, HBV_TCOV)
    tplib = tplib.clip(-17, 17)
    for i, tp in enumerate(tplib):
        p = hbv_trans2true(tp)
        tp2 = hbv_true2trans(p)
        assert allclose(tp, tp2)
        p2 = hbv_trans2true(tp2)
        assert allclose(p, p2)


def test_run1():
    sa = HBV()
    nval = 1000

    p = np.exp(np.random.normal(0, 2, size=nval))
    pe = np.ones(nval) * 5.

    inputs = np.array([p, pe]).T
    sa.allocate(inputs, 6)
    sa.initialise()
    sa.params.reset()
    sa.run()

    out = sa.outputs
    assert out.shape == (nval, 6)


@pytest.mark.parametrize("catchment", np.arange(1, 21))
def test_run2(catchment):
    if catchment == 4:
        pytest.skip("Very low flow only. Skipping.")

    warmup = 365 * 5
    sa = HBV()

    data, inputs, params = get_hbv_data(catchment)
    nval = inputs.shape[0]

    # Run hbv
    sa.allocate(inputs) #, noutputs=sa.noutputsmax)
    for nm, value in params.items():
        sa.params[nm] = value

    sa.initialise()
    sa.run()
    qsim1 = sa.outputs[:,0].copy()

    # Compare
    expected = data.loc[:, "q"].values

    start = max(warmup, np.where(expected>=0)[0].min())
    idx = np.arange(nval) > start
    qsim1 = qsim1[idx]
    expected = expected[idx]

    err = np.abs(qsim1 - expected)
    rerr = np.abs(np.arcsinh(qsim1) - np.arcsinh(expected))

    rerrmax = np.percentile(rerr, 90)
    rerr_thresh = 5e-3
    ck = rerrmax < rerr_thresh
    failmsg = f"max abs rerr = {rerrmax:0.5f} < {rerr_thresh:0.5f}"
    assert ck, failmsg


@pytest.mark.parametrize("catchment", [1])
def test_stability(catchment, allclose):
    sa = HBV()
    warmup = 365 * 6
    data, inputs, params = get_hbv_data(catchment)

    nval = inputs.shape[0]
    sa.allocate(inputs)
    sa.iend = 5

    for nm, value in params.items():
        sa.params[nm] = value

    sa.initialise_fromdata()
    sa.run()
    out1 = sa.outputs[:, 0].copy()

    sa.initialise_fromdata()
    sa.run()
    out2 = sa.outputs[:, 0].copy()
    assert allclose(out1, out2)


@pytest.mark.parametrize("catchment", np.arange(1, 21))
def test_calibrate(catchment):
    sa = HBV()
    warmup = 365 * 6
    data, inputs, params = get_hbv_data(catchment)

    calib = CalibrationHBV(objfun=ObjFunSSE())
    calib.iprint = 100

    data, inputs, params = get_hbv_data(catchment)
    nval = inputs.shape[0]

    sa.allocate(inputs)
    t0 = time.time()
    for nm, value in params.items():
        if nm in sa.params.names:
            sa.params[nm] = value
    expected = sa.params.values.copy()

    sa.initialise()
    sa.run()
    nval = sa.outputs.shape[0]
    err = np.random.uniform(-1, 1, nval)*1e-4
    obs = np.maximum(0, sa.outputs[:, 0]+err)
    ical = np.arange(nval) > warmup

    # Calibrate
    final, ofun, _, _ = calib.workflow(obs, inputs, ical=ical, \
                           maxfun=100000, ftol=1e-8)

    err = np.abs(calib.model.params.values - expected)
    ck = np.max(err) < 1e-7
    msg = f"TEST HBV CALIB {catchment:02d} :"\
          + f" max abs err = {np.max(err):3.3e}"
    print(msg)
    #assert ck

