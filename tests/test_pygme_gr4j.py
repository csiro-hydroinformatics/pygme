import os
import re
from pathlib import Path
import cProfile
import unittest
from itertools import product as prod

import pytest

import time

import numpy as np

from pygme.model import ParameterCheckValueError
from pygme.calibration import ObjFunSSE
from pygme.models.gr4j import GR4J, CalibrationGR4J
from pygme.models.gr4j import compute_PmEm, gr4j_X1_initial

import testdata

import c_pygme_models_hydromodels

import c_pygme_models_utils
UHEPS = c_pygme_models_utils.uh_getuheps()

PROFILE = True

FTESTS = Path(__file__).resolve().parent


def get_gr4j_data(catchment):
    data = testdata.read(f"GR4J_timeseries_{catchment:02d}.csv",
                         source="output", has_dates=False)
    inputs = np.ascontiguousarray(\
                    data.loc[:, ['Precip', 'PotEvap']], \
                    np.float64)
    params = testdata.read(f"GR4J_params_{catchment:02d}.csv",
                           source="output", has_dates=False)
    params = params.squeeze().values

    Pm, Em = compute_PmEm(inputs[:, 0], inputs[:, 1])

    return data, inputs, params, Pm, Em


@pytest.mark.parametrize("catchment", np.arange(1, 21))
def test_PmEm(catchment, allclose):
    data, inputs, params, Pm, Em = get_gr4j_data(catchment)
    ts = inputs[:, 0] - inputs[:, 1]
    raine = np.maximum(ts, 0)
    idx = inputs[:, 0] >= inputs[:, 1]
    Pme = np.mean(raine[idx])
    assert allclose(Pm, Pme)

    evape = np.maximum(-ts, 0)
    Eme = np.mean(evape[~idx])
    assert allclose(Em, Eme)


@pytest.mark.parametrize("catchment", np.arange(1, 21))
def test_initial(catchment, allclose):
    # Test the case where Pm=0, Em=0
    for X1 in np.logspace(0, 5, 100):
        ini = gr4j_X1_initial(0., 0., X1)
        assert allclose(ini, 0.)

    # Objective function for initial condition
    def fun(ini, Pm, Em, X1):
        ratio = ini/2.25
        isq = 1./(1+ratio**4)**0.25
        f = (1-ini**2)*Pm-ini*(2-ini)*Em-X1*ini*(1-isq);
        return f

    data, inputs, params, Pm, Em = get_gr4j_data(catchment)
    for X1 in np.logspace(0, 5, 100):
        ini = gr4j_X1_initial(Pm, Em, X1)
        f = fun(ini, Pm, Em, X1)
        assert allclose(f, 0., atol=1e-3)


def test_initial_error():
    msg = "c_pygme_models_hydromodels.gr4j_X1_initial"
    with pytest.raises(ValueError, match=msg):
        ini = gr4j_X1_initial(-1, 1, 1)

    with pytest.raises(ValueError, match=msg):
        ini = gr4j_X1_initial(1, -1, 1)

    with pytest.raises(ValueError, match=msg):
        ini = gr4j_X1_initial(1, 1, -1)


def test_print():
    gr = GR4J()
    str_gr = '%s' % gr


def test_error1():
    gr = GR4J()
    msg = "model GR4J: Expected noutputs"
    with pytest.raises(ValueError, match=msg):
        gr.allocate(np.random.uniform(0, 1, (200, 2)), 30)


def test_error2():
    gr = GR4J()
    inputs = np.random.uniform(size=(20, 3))
    msg = "model GR4J: Expected 2 inputs"
    with pytest.raises(ValueError, match=msg):
        gr.allocate(inputs, 5)
        gr.initialise()


def test_checkvalues():
    ''' Test that parameter cannot be set if not checked '''
    gr = GR4J()
    gr.X2 = -50

    msg = "X3 "
    with pytest.raises(Exception, match=msg):
        gr.X3 = 10

    with pytest.raises(Exception, match=msg):
        gr.params.values = [100., -20, 10, 0.5]


def test_uh():
    ''' Test GR4J UH '''
    gr = GR4J()
    gr.allocate(np.zeros((10, 2)))

    for x4 in np.linspace(0, 50, 100):
        # Set parameters
        gr.X1 = 400
        gr.X2 = -1
        gr.X3 = 50
        gr.X4 = x4

        ord1 = gr.params.uhs[0][1].ord
        ord2 = gr.params.uhs[1][1].ord

        assert abs(np.sum(ord1)-1) < UHEPS * 2
        assert abs(np.sum(ord2)-1) < UHEPS * 2


def test_run_dimensions():
    ''' Allocate GR4J '''
    gr = GR4J()
    nval = 1000

    p = np.exp(np.random.normal(0, 2, size=nval))
    pe = np.ones(nval) * 5.

    inputs = np.array([p, pe]).T
    gr.allocate(inputs, 11)
    gr.initialise()
    gr.run()

    out = gr.outputs
    assert out.shape == (nval, 11)


@pytest.mark.parametrize("catchment", np.arange(1, 21))
def test_run_against_data(catchment, allclose):
    warmup = 365 * 5
    gr = GR4J()
    data, inputs, params, Pm, Em = get_gr4j_data(catchment)
    # Run gr4j
    gr.allocate(inputs, 11)
    gr.params.values = params

    # .. initiase to same values than IRSTEA run ...
    # Estimate initial states based on first two state values
    s0 = data.loc[0, ['Prod', 'Rout']].values
    s1 = data.loc[1, ['Prod', 'Rout']].values
    sini = 2*s0-s1
    gr.initialise(states=sini)

    gr.run()

    # Compare
    idx = np.arange(inputs.shape[0]) > warmup
    sim = gr.outputs[:, [0, 6, 7]].copy()
    expected = data.loc[:, ['Qsim', 'QD', 'QR']].values

    err = np.abs(sim[idx, :] - expected[idx, :])

    # Sensitivity to initial conditionos
    s1 = [0]*2
    s2 = [gr.params.X1, gr.params.X3]
    warmup_ideal, sim0, sim1 = gr.inisens(s1, s2)

    # Special criteria
    # 5 values with difference greater than 1e-5
    # max diff lower than 5e-4
    def fun(x):
        return np.sum(x > 1e-5), np.max(x)

    cka = np.array([fun(err[:, k]) for k in range(err.shape[1])])
    ck = np.all((cka[:, 0] < 5) & (cka[:, 1] < 1e-4))

    msg = f"TEST GR4J SIM {catchment:2d} : "\
          + f"crit={ck} err={np.max(err):3.3e}"\
          + f" warmup={warmup_ideal}"
    print(msg)
    assert ck


@pytest.mark.parametrize("catchment", np.arange(1, 21))
def test_initialisation(catchment, allclose):
    warmup = 365 * 5
    gr = GR4J()
    data, inputs, params, Pm, Em = get_gr4j_data(catchment)

    for X1, X3 in prod(np.logspace(0, 4, 5), np.logspace(0, 4, 5)):
        gr.params.X1 = X1
        gr.params.X3 = X3
        gr.initialise_fromdata(Pm, Em)
        ini = gr4j_X1_initial(Pm, Em, X1)
        assert allclose(gr.states.values[0], ini*X1)
        assert allclose(gr.states.values[1], 0.3*X3)

        gr.initialise_fromdata()
        assert allclose(gr.states.values[0], 0.5*X1)
        assert allclose(gr.states.values[1], 0.3*X3)


@pytest.mark.parametrize("catchment", np.arange(1, 21))
def test_calibration_against_itself(catchment, allclose):
    warmup = 365 * 5
    gr = GR4J()
    data, inputs, params, Pm, Em = get_gr4j_data(catchment)

    # Run gr first
    gr.allocate(inputs, 1)
    gr.params.values = params
    gr.initialise_fromdata(Pm, Em)
    gr.run()

    # Add error to outputs
    err = np.random.uniform(-1, 1, gr.outputs.shape[0]) * 1e-4
    obs = gr.outputs[:,0].copy()+err

    # Wrapper function for profiling
    t0 = time.time()
    calib = CalibrationGR4J(objfun=ObjFunSSE(), Pm=Pm, Em=Em)
    def profilewrap(outputs):
        final, ofun, _, _ = calib.workflow(obs, inputs, \
                                    maxfun=100000, ftol=1e-8)
        outputs.append(final)
        outputs.append(ofun)

    # Run profiler
    if PROFILE and catchment == 11:
        pstats = FTESTS / f"gr4j_calib{catchment:02d}.pstats"
        outputs = []
        prof = cProfile.runctx('profilewrap(outputs)', globals(), \
                    locals(), filename=pstats)
        final = outputs[0]
        ofun = outputs[1]
    else:
        outputs = []
        profilewrap(outputs)
        final = outputs[0]
        ofun = outputs[1]

    t1 = time.time()
    dt = (t1-t0) / len(obs)*365.25

    # Test error on parameters
    err = np.abs(final-params)
    imax = np.argmax(err)
    ck = np.allclose(params, final, rtol=1e-3, atol=1e-3)

    msg = f"TEST GR4J CALIB ITSELF {catchment:02d} :"\
          + f" PASSED?{ck} err[X{imax + 1}]={err[imax]:3.3e}"\
          + f" dt={dt:3.3e} sec/yr"
    print(msg)
    assert ck


@pytest.mark.parametrize("catchment", [11])
def test_calibrate_fixed(catchment, allclose):
    gr = GR4J()
    warmup = 365*6
    calib1 = CalibrationGR4J(objfun=ObjFunSSE())
    calib2 = CalibrationGR4J(objfun=ObjFunSSE(), \
                    fixed={'X1':1000, 'X4':10})

    data, inputs, expected, Pm, Em = get_gr4j_data(catchment)

    # Run gr first
    gr = calib1.model
    gr.allocate(inputs, 1)
    gr.params.values = expected
    gr.initialise()
    gr.run()

    # Calibrate
    err = np.random.uniform(-1, 1, gr.outputs.shape[0]) * 1e-4
    obs = gr.outputs[:,0].copy()+err

    final1, ofun1, _, _ = calib1.workflow(obs, inputs,
                                          maxfun=100000,
                                          ftol=1e-8)

    final2, ofun2, _, _ = calib2.workflow(obs, inputs,
                                          maxfun=100000,
                                          ftol=1e-8)

    # Check one calibration works as expected
    assert allclose(final1, expected, atol=1e-1, rtol=0.)

    # Check the other one returns fixed parameters
    assert allclose(final2[[0, 3]], [1000, 10])

