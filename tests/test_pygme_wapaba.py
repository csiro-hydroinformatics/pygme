import os, sys
import re
from pathlib import Path

import pytest

from timeit import Timer
import time

import numpy as np

from pygme.models.wapaba import WAPABA, CalibrationWAPABA
from pygme.calibration import ObjFunSSE

import warnings

import testdata

def test_print():
    wp = WAPABA()
    s = f"{wp}"

def test_wapaba_dumb():
    wp = WAPABA()

    nval = 100
    p = np.exp(np.random.normal(0, 2, size=nval))
    pe = np.ones(nval) * 5.
    inputs = np.column_stack([p, pe])
    wp.allocate(inputs, 9)

    wp.ALPHA1 = 2.
    wp.ALPHA2 = 2.
    wp.BETA = 0.5
    wp.SMAX = 100
    wp.INVK = 0.5
    wp.initialise()

    wp.run()


def test_run(allclose):
    """ Compare GR2M simulation with test data """
    warmup = 12 * 12
    wp = WAPABA()

    for i in range(20):
        if i in [2, 9, 14]:
            warnings.warn("Skipping test {0} for GR2m".format(i+1))
            continue

        data = testdata.read("GR2M_timeseries_{0:02d}.csv".format(i+1), \
                                source="output", has_dates=False)
        params = [2., 2., 0.5, 100, 0.5]

        inputs = np.ascontiguousarray(\
                        data.loc[:, ["Precip", "PotEvap"]], \
                        np.float64)
        nval = len(inputs)

        # Run gr2m
        wp.allocate(inputs, 10)
        wp.params.values = params
        wp.initialise_fromdata()
        wp.run()


def test_wapaba_calib(allclose):
    """ Test gr2m calibration """
    i = 0

    # Get data
    data = testdata.read("GR2M_timeseries_{0:02d}.csv".format(i+1), \
                            source="output", has_dates=False)

    inputs = np.ascontiguousarray(\
                    data.loc[:, ["Precip", "PotEvap"]], \
                    np.float64)
    nval = len(inputs)
    wp = WAPABA()
    wp.allocate(inputs)

    # Calibration object
    calib = CalibrationWAPABA(objfun=ObjFunSSE(), \
                                    warmup=120, \
                                    nparamslib=10000)
    # Sample parameters
    nsamples = 10
    samples = calib.paramslib[:nsamples]

    # loop through parameters
    for i, expected in enumerate(samples):

        # Generate obs
        wp.params.values = expected
        wp.initialise()
        wp.run()

        # Produce theoretical observation
        # with small error corruption
        err = np.random.uniform(-1, 1, wp.outputs.shape[0]) * 1e-4
        obs = np.maximum(0., wp.outputs[:,0].copy()+err)

        # Calibrate
        final, ofun, _, _ = calib.workflow(obs, inputs, \
                                    maxfun=100000, ftol=1e-8)
        # Test
        warmup = calib.warmup
        sim = calib.model.outputs[:, 0]
        rerr = np.abs(obs[warmup:]-sim[warmup:])/(1+obs[warmup:])*100
        rerrmax = np.percentile(rerr, 90) # leaving aside 10% of the series
        assert rerrmax < 1e-2

        #params = calib.model.params.values
        #err = np.abs(params-expected)

