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


def test_uh():
    pytest.skip("WIP")
    sa = HBV()
    sa.allocate(np.zeros((10, 2)))

    for Lag in np.linspace(0, 50, 100):
        sa.params.reset()
        sa.params.Lag = Lag

        ordu = sa.params.uhs[0][1].ord
        assert abs(np.sum(ordu)-1) < UHEPS * 1


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


def test_run2():
    pytest.skip("WIP")
    rerr_thresh = 8e-2
    warmup = 365*11
    sa = HBV()

    fp = FHERE / "hbv" / "HBV_params.csv"
    params = pd.read_csv(fp, index_col="id")
    params = params.sort_index()

    for i, param in params.iterrows():
        fts = FHERE / "hbv" / f"HBV_timeseries_{i:02d}.csv"
        data = pd.read_csv(fts)
        inputs = np.ascontiguousarray(data.iloc[:, [1, 2]], np.float64)
        nval = inputs.shape[0]

        # Run hbv [block]
        sa.allocate(inputs)
        for nm, value in param.items():
            if nm in sa.params.names:
                sa.params[nm] = value

        sa.params.Lag = 1-sa.params.Lag

        sa.initialise()
        sa.run()
        qsim1 = sa.outputs[:,0].copy()

        # Compare
        expected = data.loc[:, "hbv[mm/d]"].values

        start = max(warmup, np.where(expected>=0)[0].min())
        idx = np.arange(nval) > start
        qsim1 = qsim1[idx]
        expected = expected[idx]

        err = np.abs(qsim1 - expected)
        rerr = err/(1e-1+expected)*100
        ck = rerr.max() < rerr_thresh
        failmsg = f"TEST {i+1} : max abs rerr = " +\
            f"{rerr.max():0.5f} < {rerr_thresh:0.5f}"
        assert ck, failmsg



def test_calibrate():
    return
    sa = HBV()
    warmup = 365*6

    calib = CalibrationHBV(objfun=ObjFunSSE())
    calib.iprint = 100
    sa = HBV()

    fp = FHERE / "hbv" / "HBV_params.csv"
    params = pd.read_csv(fp, index_col="id")
    params = params.sort_index()

    for i, param in params.iterrows():
        fts = FHERE / "hbv" / f"HBV_timeseries_{i:02d}.csv"
        data = pd.read_csv(fts)
        inputs = np.ascontiguousarray(data.iloc[:, [1, 2]], np.float64)
        nval = inputs.shape[0]

        # Run hbv to define obs
        sa.allocate(inputs)
        t0 = time.time()
        for nm, value in param.items():
            if nm in sa.params.names:
                sa.params[nm] = value
        expected = sa.params.values.copy()

        sa.params.Lag = 1-sa.params.Lag
        sa.initialise()
        sa.run()
        nval = sa.outputs.shape[0]
        err = np.random.uniform(-1, 1, nval)*1e-4
        obs = np.maximum(0, sa.outputs[:, 0]+err)
        ical = np.arange(nval)>warmup

        # Calibrate
        #final, ofun, _, _ = calib.workflow(obs, inputs, ical=ical, \
        #                       maxfun=100000, ftol=1e-8)
        calib.allocate(obs, inputs)
        calib.ical = ical

        start, _, ofun_explore = calib.explore(iprint=500)
        calib.calparams.truevalues = start
        tstart = calib.calparams.values
        final, ofun_final, outputs_final = calib.fit(tstart, \
                                    iprint=50, \
                                    optimizer=fmin_bfgs) #, \
                                    #maxfun=100000) #, ftol=1e-8)

        # Not tested!
        #err = np.abs(calib.model.params.values - expected)
        #ck = np.max(err) < 1e-7

        #sim = calib.model.outputs[:, 0]
        #err = sim[ical]-obs[ical]

        #import matplotlib.pyplot as plt
        #plt.close("all")
        #plt.plot(obs)
        #plt.plot(sim)
        #plt.show()
        #import pdb; pdb.set_trace()


        #print(("\t\tTEST CALIB {0:02d} : max abs err = {1:3.3e}" +
        #        " neval= {2} + {3}").format( \
        #            i+1, np.max(err), ieval1, ieval2))

        #assert ck

