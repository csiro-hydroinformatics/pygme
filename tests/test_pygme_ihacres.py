import os, sys
import re
from pathlib import Path

import pytest

from timeit import Timer
import time

import numpy as np
import pandas as pd

from pygme.models.ihacres import IHACRES, CalibrationIHACRES, \
                                IHACRES_TMEAN, IHACRES_TCOV, \
                                ihacres_trans2true, ihacres_true2trans

from pygme.calibration import ObjFunSSE

from hydrodiy.stat import sutils

np.random.seed(5446)

import warnings

import testdata

def test_print():
    ihc = IHACRES()
    s = f"{ihc}"

def test_ihacres_dumb(allclose):
    ihc = IHACRES()

    nval = 100
    p = np.exp(np.random.normal(0, 2, size=nval))
    pe = np.ones(nval) * 5.
    inputs = np.column_stack([p, pe])
    ihc.allocate(inputs, ihc.noutputsmax)

    ihc.f = 0.7
    ihc.d = 200
    ihc.delta = 0.5
    assert allclose(ihc.params.values, [0.7, 200, 0.5])

    ihc.initialise()
    ihc.run()


def test_params_transform(allclose):
    nparamslib = 10000
    model = IHACRES()
    tplib = sutils.lhs_norm(nparamslib, IHACRES_TMEAN, IHACRES_TCOV)
    # to make sure that transform parameters do not hit bounds
    tplib[:, -2:] = np.maximum(tplib[:, -2:], -4.6)
    for i, tp in enumerate(tplib):
        p = ihacres_trans2true(tp)
        tp2 = ihacres_true2trans(p)
        assert allclose(tp, tp2)

        p2 = ihacres_trans2true(tp2)
        assert allclose(p, p2)


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
        ihc.allocate(inputs, ihc.noutputsmax)
        ihc.f = params.f
        ihc.d = params.d
        ihc.delta = 0.5
        ihc.initialise_fromdata()
        ihc.run()

        outputs = ihc.to_dataframe().loc[:, ["U", "M", "ET"]]

        expected = data.loc[:, ["U", "CMD", "ET"]].iloc[:nval]
        assert np.allclose(outputs, expected)


#def test_run_shapes(allclose):
#    """ Compare IHACRES simulation using different shapes """
#    warmup = 12 * 12
#
#    ihcs = [IHACRES(sh) for sh in [0, 1, 1.5, 2, 5, 10]]
#
#    for i in range(20):
#        data = testdata.read("IHACRES_timeseries_{0:02d}.csv".format(i+1), \
#                                source="output", has_dates=False)
#
#        params = testdata.read('IHACRES_params_{0:02d}.csv'.format(i+1), \
#                                source='output', has_dates=False)
#
#        inputs = np.ascontiguousarray(\
#                        data.loc[:, ["P", "E"]], \
#                        np.float64)
#        nval = len(inputs)
#
#        # Run ihacres
#        sims = {}
#        for model in ihcs:
#            model.allocate(inputs, model.noutputsmax)
#            model.f = params.f
#            model.d = params.d
#            model.initialise_fromdata()
#            model.run()
#            sims[f"sh{model.shape:0.1f}"] = model.outputs[:, 0]
#
#        sims = pd.DataFrame(sims)
#
#        import pdb; pdb.set_trace()



def test_ihacres_calib(allclose):
    """ Test ihacres calibration """
    # Get data
    i = 0
    data = testdata.read("IHACRES_timeseries_{0:02d}.csv".format(i+1), \
                            source="output", has_dates=False)

    inputs = np.ascontiguousarray(data.loc[:, ["P", "E"]], np.float64)
    nval = len(inputs)
    ihc = IHACRES()
    ihc.allocate(inputs, ihc.noutputsmax)

    # Calibration object
    calib = CalibrationIHACRES(objfun=ObjFunSSE(), \
                                    warmup=120, \
                                    nparamslib=10000)
    # Sample parameters
    nsamples = 10
    samples = calib.paramslib[:nsamples]

    # loop through parameters
    for i, expected in enumerate(samples):
        print(f"Calib test {i+1}/{nsamples}")

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
        try:
            assert rerrmax < 5e-2
        except:
            import matplotlib.pyplot as plt
            plt.plot(obs)
            plt.plot(sim)
            plt.show()
            import pdb; pdb.set_trace()

        #params = calib.model.params.values
        #err = np.abs(params-expected)

