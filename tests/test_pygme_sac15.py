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
from pygme.models.sac15 import SAC15, CalibrationSAC15, \
                            SAC15_TMEAN, SAC15_TCOV, \
                            sac15_trans2true, sac15_true2trans

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
    sa = SAC15()
    LOGGER.info(str(sa))


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


def test_params_transform(allclose):
    nparamslib = 10000
    model = SAC15()
    tplib = sutils.lhs_norm(nparamslib, SAC15_TMEAN, SAC15_TCOV)
    tplib = tplib.clip(-17, 17)
    for i, tp in enumerate(tplib):
        p = sac15_trans2true(tp)
        tp2 = sac15_true2trans(p)
        assert allclose(tp, tp2)
        p2 = sac15_trans2true(tp2)
        assert allclose(p, p2)


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
        for nm, value in param.items():
            if nm in sa.params.names:
                sa.params[nm] = value

        sa.params.Lag = 1-sa.params.Lag

        sa.initialise()
        sa.run()
        qsim1 = sa.outputs[:,0].copy()

        # Compare
        expected = data.loc[:, "sac15[mm/d]"].values

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
    sa = SAC15()
    warmup = 365*6

    calib = CalibrationSAC15(objfun=ObjFunSSE())
    calib.iprint = 100
    sa = SAC15()

    fp = FHERE / "sac15" / "SAC15_params.csv"
    params = pd.read_csv(fp, index_col="id")
    params = params.sort_index()

    for i, param in params.iterrows():
        fts = FHERE / "sac15" / f"SAC15_timeseries_{i:02d}.csv"
        data = pd.read_csv(fts)
        inputs = np.ascontiguousarray(data.iloc[:, [1, 2]], np.float64)
        nval = inputs.shape[0]

        # Run sac15 to define obs
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

