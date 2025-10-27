import os
import re
import pytest
import itertools

import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from pygme.models.lagroute import LagRoute, CalibrationLagRoute, NORDMAXMAX
from pygme.calibration import Calibration, CalibParamsVector, ObjFunSSE


import c_pygme_models_utils
UHEPS = c_pygme_models_utils.uh_getuheps()

import testdata

np.random.seed(0)


def test_print():
    lr = LagRoute()
    str_lr = '%s' % lr


def test_run():
    ierr_id = ''
    lr = LagRoute()

    inputs = np.zeros((20, 1))
    inputs[1,0] = 100
    lr.allocate(inputs, 2)
    lr.initialise()
    lr.run()


def test_error1():
    lr = LagRoute()
    msg = 'model LagRoute: Expected noutputs in .1, 4.'
    with pytest.raises(ValueError, match=msg):
        inputs = np.zeros((20, 1))
        lr.allocate(inputs, 30)


def test_error2():
    lr = LagRoute()
    msg = 'model LagRoute: Expected 1 inputs'
    with pytest.raises(ValueError, match=msg):
        inputs = np.random.uniform(size=(20, 3))
        lr.allocate(inputs, 2)
        lr.initialise()


def test_uh1():
    lr = LagRoute()
    for u, a in itertools.product(np.linspace(0, 10, 20), \
            np.linspace(0, 1, 20)):
        lr.params.values = [u, a]
        ord = lr.params.uhs[0][1].ord
        ck = abs(np.sum(ord)-1) < UHEPS
        assert ck


def test_uh2():
    lr = LagRoute()

    # Set config
    cfg = lr.config
    cfg.timestep = 86400 # daily model
    cfg.length = 86400 # 86.4 km reach
    cfg.flowref = 1 # qstar = 1 m3/s
    cfg.storage_expon = 1 # linear model

    alpha = 1.
    Umin = lr.params.mins[0]
    Umax = lr.params.maxs[0]

    # Set uh
    for U in np.linspace(Umin, Umax, 100):

        lr.params.U = U
        lr.params.alpha = alpha

        ord = lr.params.uhs[0][1].ord
        ck = abs(np.sum(ord)-1) < 1e-5
        assert ck

        tau = alpha * cfg.length * U
        k = int(tau/cfg.timestep)
        w = tau/cfg.timestep - k
        ck = abs(ord[k]-1+w) < 1e-5
        assert ck


def test_max_invv():
    lr = LagRoute()
    # Test if maximum uh length can be set
    length, timestep = 100e3, 3600 # 100km / 1hr
    lr.config.length = length
    lr.config.timestep = timestep

    # Umax = NORDMAXMAX/length*timestep
    lr.alpha = 1.
    # .. creates an error if the max ordinate is not controlled
    lr.U = (NORDMAXMAX+1)/length*timestep


def test_massbalance():

    nval = 1000
    q1 = np.exp(np.random.normal(0, 2, size=nval))
    inputs = np.ascontiguousarray(q1[:,None])

    lr = LagRoute()

    # Set config
    cfg = lr.config
    cfg.timestep = 86400 # daily model
    cfg.length = 86400 # 86.4 km reach
    cfg.flowref = 50 # qstar = 1 m3/s

    # Set outputs
    lr.allocate(inputs, 4)

    print("")
    for theta2 in [1, 2]:
        lr.config.storage_expon = theta2

        # Run
        UU = np.linspace(0.1, 20, 20)
        aa = np.linspace(0., 1., 20)
        dta = 0
        count = 0

        for U, alpha in itertools.product(UU, aa):

            t0 = time.time()

            lr.params.U = U
            lr.params.alpha = alpha
            lr.initialise()
            lr.run()

            t1 = time.time()
            dta += 1000 * (t1-t0) / nval * 365.25

            v0 = 0
            vr = lr.outputs[-1, 2]
            v1 = lr.outputs[-1, 3]
            si = np.sum(inputs) * cfg.timestep
            so = np.sum(lr.outputs[:,0]) * cfg.timestep

            B = si - so - v1 - vr + v0
            ck = abs(B/so) < 1e-10

            assert ck

        dta /= (len(UU) * len(aa))
        print(('\t\ttheta2={0} - Average runtime'+\
                ' = {1:.5f} ms/yr').format( \
                    theta2, dta))


def test_lagroute_lag():
    ''' Test lagroute with pure lag '''
    nval = 1000
    q1 = np.exp(np.random.normal(0, 2, size=nval))
    inputs = np.ascontiguousarray(q1[:,None])

    lr = LagRoute()

    # Set configuration
    cfg = lr.config
    cfg.timestep = 86400 # daily model
    cfg.length = 86400 # 86.4 km reach
    cfg.flowref = 50 # qstar = 1 m3/s

    # Set outputs
    lr.allocate(inputs)

    # Run
    for U in range(1, 11):
        lr.params.U = U
        lr.params.alpha = 1
        lr.initialise()
        lr.run()

        err = np.abs(lr.outputs[U:,0] - inputs[:-U, 0])

        ck = np.max(err) < 1e-10
        assert ck


def test_calibrate_against_it():
    ''' Calibrate lag route against a simulation
        with known parameters '''
    lr = LagRoute()
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

        # Run lagroute first
        lr.allocate(inputs, 1)
        lr.params.values = params
        lr.initialise()
        lr.run()

        # Add error to outputs
        err = np.random.uniform(-1, 1, lr.outputs.shape[0]) * 1e-4
        obs = lr.outputs[:,0].copy()+err

        # Wrapper function for profiling

        calib = CalibrationLagRoute(objfun=objfun)
        final, ofun, _, _ = calib.workflow(obs, inputs, \
                                    maxfun=100000, ftol=1e-8)

        # Test if error on outputs
        warmup = calib.warmup
        sim = calib.model.outputs[:, 0]
        rerr = np.arcsinh(obs[warmup:]) - np.arcsinh(sim[warmup:])
        rerrmax = np.percentile(rerr, 90) # leaving aside 10% of the series
        assert (rerrmax < 2e-3)
