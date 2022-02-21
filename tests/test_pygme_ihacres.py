import os, sys
import re
from pathlib import Path

import pytest

from timeit import Timer
import time

import numpy as np

from pygme.models.ihacres import IHACRES, CalibrationIHACRES
from pygme.calibration import ObjFunSSE

import warnings

import testdata

def test_print():
    ihc = IHACRES()
    s = f"{ihc}"

def test_ihacres_dumb():
    ihc = IHACRES()

    nval = 100
    p = np.exp(np.random.normal(0, 2, size=nval))
    pe = np.ones(nval) * 5.
    inputs = np.column_stack([p, pe])
    ihc.allocate(inputs, 4)

    ihc.f = 0.7
    ihc.e = 0.166
    ihc.d = 200
    ihc.initialise()
    ihc.run()


def test_run(allclose):
    """ Compare IHACRES simulation with test data """
    warmup = 12 * 12
    ihc = IHACRES()

    for i in range(20):
        data = testdata.read("IHACRES_timeseries_{0:02d}.csv".format(i+1), \
                                source="output", has_dates=False)

        params = testdata.read('IHACRES_params_{0:02d}.csv'.format(i+1), \
                                source='output', has_dates=False)

        inputs = np.ascontiguousarray(\
                        data.loc[:, ["P", "E"]], \
                        np.float64)
        nval = len(inputs)

        # Run ihacres
        ihc.allocate(inputs, 4)
        ihc.params.values = [params[n][0] for n in ihc.params.names]
        ihc.initialise_fromdata()
        ihc.run()
        outputs = ihc.outputs_dataframe().loc[:, ["Q", "M", "ET"]]

        expected = data.loc[:, ["U", "CMD", "ET"]].iloc[:nval]
        assert np.allclose(outputs, expected)



def test_ihacres_calib(allclose):
    """ Test ihacres calibration """
    # Get data
    i = 0
    data = testdata.read("IHACRES_timeseries_{0:02d}.csv".format(i+1), \
                            source="output", has_dates=False)

    inputs = np.ascontiguousarray(data.loc[:, ["P", "E"]], np.float64)
    nval = len(inputs)
    ihc = IHACRES()
    ihc.allocate(inputs)

    # Calibration object
    calib = CalibrationIHACRES(objfun=ObjFunSSE(), \
                                    warmup=120, \
                                    nparamslib=10000)
    # Sample parameters
    nsamples = 10
    samples = calib.paramslib[:nsamples]

    # loop through parameters
    for i, expected in enumerate(samples):

        # Generate obs
        ihc.params.values = expected
        ihc.initialise()
        ihc.run()

        # Produce theoretical observation
        # with small error corruption
        err = np.random.uniform(-1, 1, ihc.outputs.shape[0]) * 1e-4
        obs = np.maximum(0., ihc.outputs[:,0].copy()+err)

        # Calibrate
        final, ofun, _, _ = calib.workflow(obs, inputs, \
                                    maxfun=100000, ftol=1e-8)
        # Test
        warmup = calib.warmup
        sim = calib.model.outputs[:, 0]
        rerr = np.abs(obs[warmup:]-sim[warmup:])/(1+obs[warmup:])*100
        rerrmax = np.percentile(rerr, 90) # leaving aside 10% of the series
        assert rerrmax < 2e-2

        #params = calib.model.params.values
        #err = np.abs(params-expected)

