import os
import re
import pytest
from pathlib import Path

import time
import math

import numpy as np
import pandas as pd

from pygme import calibration
from pygme.models.sac15 import SAC15, CalibrationSAC15

import c_pygme_models_utils
UHEPS = c_pygme_models_utils.uh_getuheps()

filename = Path(__file__).resolve()
FHERE = filename.parent

def test_print():
    sa = SAC15()
    print(sa)


def test_error1():
    sa = SAC15()
    msg = "model SAC15: Expected noutputs in"
    with pytest.raises(ValueError, match=msg):
        sa.allocate(np.random.uniform(0, 1, (200, 2)), 30)


def test_error2():
    sa = SAC15()
    inputs = np.random.uniform(size=(20, 3))
    msg = "model SAC15: Expected 2 inputs, got"
    with pytest.raises(ValueError, match=msg):
        sa.allocate(inputs, 5)
        sa.initialise()


def test_uh():
    sa = SAC15()
    sa.allocate(np.zeros((10, 2)))

    for Lag in np.linspace(0, 50, 100):
        sa.params.reset()
        sa.params.Lag = Lag

        ordu = sa.params.uhs[0][1].ord
        assert abs(np.sum(ordu)-1) < UHEPS * 1


def test_run1():
    sa = SAC15()
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


def test_run2():
    rerr_thresh = 8e-2
    warmup = 365*11
    sa = SAC15()

    fp = FHERE / "sac15" / "SAC15_params.csv"
    params = pd.read_csv(fp, index_col="id")
    params = params.sort_index()

    for i, param in params.iterrows():
        fts = FHERE / "sac15" / f"SAC15_timeseries_{i:02d}.csv"
        data = pd.read_csv(fts)
        inputs = np.ascontiguousarray(data.iloc[:, [1, 2]], np.float64)
        nval = inputs.shape[0]

        # Run sac15 [block]
        sa.allocate(inputs)
        t0 = time.time()
        for nm, value in param.items():
            if nm in sa.params.names:
                sa.params[nm] = value

        sa.params.Lag = 1-sa.params.Lag

        sa.initialise()
        sa.run_as_block = True
        sa.run()

        qsim1 = sa.outputs[:,0].copy()
        t1 = time.time()
        dta1 = 1000 * (t1-t0)
        dta1 /= len(qsim1)/365.25

        # Compare
        expected = data.loc[:, "sac15[mm/d]"].values

        start = max(warmup, np.where(expected>=0)[0].min())
        idx = np.arange(nval) > start
        qsim1 = qsim1[idx]
        expected = expected[idx]

        err = np.abs(qsim1 - expected)
        rerr = err/(1e-2+expected)*100
        ck = np.max(rerr) < rerr_thresh

        failmsg = f"TEST {i+1} : max abs rerr = " +\
            f"{rerr.max():0.5f} < {rerr_thresh:0.5f}"
        assert ck, failmsg



def test_calibrate():
    return
    sa = SAC15()
    warmup = 365*6

    calib = CalibrationSAC15()


    fp = FHERE / "sac15" / "SAC15_params.csv"
    params = np.loadtxt(fp, delimiter=",")

    for i in range(params.shape[0]):

        fts = FHERE / "data" / f"SAC15_timeseries_{i+1:02d}.csv"
        data = np.loadtxt(fts, delimiter=",")
        inputs = Matrix.from_data("inputs",
                np.ascontiguousarray(data[:, [1, 2]], np.float64))

        # Run sa first
        params_expected = params[i, [2, 0, 1, 3]]
        sa = calib.model
        sa.allocate(inputs, 1)
        sa.params = params_expected
        sa.initialise()
        sa.run()
        obs = Matrix.from_data("obs", sa.outputs[:,0].copy())

        # Calibrate
        calib.setup(obs, inputs)

        start, explo, explo_ofun = calib.explore()
        ieval1 = calib.ieval

        final, _, _ = calib.fit(start, ftol=1e-8)
        ieval2 = calib.ieval - ieval1

        err = np.abs(calib.model.params - params_expected)
        ck = np.max(err) < 1e-7

        print(("\t\tTEST CALIB {0:02d} : max abs err = {1:3.3e}" +
                " neval= {2} + {3}").format( \
                    i+1, np.max(err), ieval1, ieval2))

        assert ck

