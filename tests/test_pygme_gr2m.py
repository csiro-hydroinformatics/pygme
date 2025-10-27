from pathlib import Path
import re
import pytest

from timeit import Timer
import time

import numpy as np
from pygme.models.gr2m import GR2M, CalibrationGR2M
from pygme.calibration import ObjFunSSE

import warnings

import testdata

np.random.seed(5446)

source_file = Path(__file__).resolve()
FTEST = source_file.parent

def get_gr2m_data(catchment):
    data = testdata.read(f"GR2M_timeseries_{catchment:02d}.csv", \
                            source="output", has_dates=False)
    params = testdata.read(f"GR2M_params_{catchment:02d}.csv", \
                            source="output", has_dates=False)
    return data, params


def test_print():
    gr = GR2M()
    str_gr = str(gr)


def test_rcapacity():
    for r in [10, 60, 100, 1000]:
        gr = GR2M(r)
        assert gr.Rcapacity == r


def test_gr2m_dumb():
    gr = GR2M()

    nval = 100
    p = np.exp(np.random.normal(0, 2, size=nval))
    pe = np.ones(nval) * 5.
    inputs = np.concatenate([p[:,None], pe[:, None]], axis=1)
    gr.allocate(inputs, 12)

    gr.X1 = 400
    gr.X2 = 0.9
    gr.initialise()

    gr.run()
    df = gr.to_dataframe()


@pytest.mark.parametrize("catchment", np.arange(1, 21))
def test_run_rcapacity(catchment, allclose):
    if catchment in [2, 9, 14]:
        pytest.skip(f"Skipping test {catchment + 1} for GR2m")

    warmup = 12 * 12
    gr = GR2M()
    gr100 = GR2M(100) # Test GR2M with different R capacity

    data, params = get_gr2m_data(catchment)
    inputs = np.ascontiguousarray(\
                    data.loc[:, ["Precip", "PotEvap"]], \
                    np.float64)

    # Run gr4j
    gr.allocate(inputs, 10)
    gr.params.values = params

    gr100.allocate(inputs, 10)
    gr100.params.values = params

    R2 = gr.to_dataframe().R2
    expected = R2**2/(100+R2)
    assert allclose(gr100.outputs[:, 0], expected)


@pytest.mark.parametrize("catchment", np.arange(1, 21))
def test_run_against_data(catchment):
    if catchment in [3, 10, 15]:
        pytest.skip(f"Skipping test {catchment} for GR2m")

    warmup = 12 * 12
    gr = GR2M()

    data, params = get_gr2m_data(catchment)
    inputs = np.ascontiguousarray(\
                    data.loc[:, ["Precip", "PotEvap"]], \
                    np.float64)

    # Run gr4j
    gr.allocate(inputs, 10)
    gr.params.values = params

    # .. initiase to same values than IRSTEA run ...
    # Estimate initial states based on first two state values
    #s0 = data[0, [6, 7]]
    #s1 = data[1, [6, 7]]
    #sini = 2*s0-s1

    # Other initial condition from GR2m Excel
    sini = [gr.params.X1/2, 30]
    gr.initialise(states=sini)
    gr.run()

    # Compare
    idx = np.arange(inputs.shape[0]) > warmup
    #sim = gr.outputs[:, [0, 3, 6, 9]].copy()
    #expected = data[:, [8, 5, 4, 2]]
    sim = gr.outputs[:, 0][:, None].copy()
    expected = data.loc[:, "Qsim2"].values[:, None]
    err = np.abs(sim[idx, :] - expected[idx, :])

    # Sensitivity to initial conditionos
    s1 = [0]*2
    s2 = [gr.params.X1, 60]
    warmup_ideal, sim0, sim1 = gr.inisens(s1, s2)

    # Special criteria
    # 5 values with difference greater than 1e-5
    # max diff lower than 5e-4
    def fun(x):
        return np.sum(x > 1e-5), np.max(x)

    cka = np.array([fun(err[:, k]) for k in range(err.shape[1])])
    ck = np.all((cka[:, 0] < 15) & (cka[:, 1] < 1e-4))

    msg = f"TEST GR2M SIM {catchment:2d} : crit={ck}"\
          + f" err={np.max(err):3.3e} warmup={warmup_ideal}"
    print(msg)
    assert ck


@pytest.mark.parametrize("catchment", [1])
def test_gr2m_calib(catchment):
    data, params = get_gr2m_data(catchment)
    inputs = np.ascontiguousarray(\
                    data.loc[:, ["Precip", "PotEvap"]], \
                    np.float64)

    gr = GR2M()
    gr.allocate(inputs)

    # Calibration object
    calib = CalibrationGR2M(objfun=ObjFunSSE(), nparamslib=1000)

    # Sample parameters
    nsamples = 10
    samples = calib.paramslib[:nsamples]

    # loop through parameters
    print("\n")
    for i, expected in enumerate(samples):
        # Generate obs
        gr.params.values = expected
        gr.initialise()
        gr.run()

        # Produce theoretical observation
        # with error corruption
        err = np.random.uniform(-1, 1, gr.outputs.shape[0]) * 1e-3
        obs = gr.outputs[:,0].copy()+err

        # Calibrate
        final, ofun, _, _ = calib.workflow(obs, inputs, \
                                    maxfun=100000, ftol=1e-8)

        # Test
        err = np.abs(final-expected)
        imax = np.argmax(err)
        ck = np.allclose(expected, final, rtol=5e-2, atol=1e-3)
        msg = f"TEST GR2M CALIB {catchment:02d} : PASSED?{ck:5}"\
              + f" err[X{imax + 1}] = {err[imax]:3.3e}"
        print(msg)
        assert ck


@pytest.mark.parametrize("catchment", [1])
def test_gr2m_calib_fixed(catchment, allclose):
    """ Test GR2M calibration with fixed parameters """
    data, params = get_gr2m_data(catchment)
    expected = testdata.read(f"GR2M_params_{catchment:02d}.csv", \
                            source="output", has_dates=False)
    expected = expected.values[:, 0]

    inputs = np.ascontiguousarray(\
                    data.loc[:, ["Precip", "PotEvap"]], \
                    np.float64)

    gr = GR2M()
    gr.allocate(inputs, 1)

    # Calibration object
    calib1 = CalibrationGR2M(objfun=ObjFunSSE())
    calib2 = CalibrationGR2M(objfun=ObjFunSSE(), fixed={"X1":200})

    # Generate obs
    gr.params.values = expected
    gr.initialise()
    gr.run()

    # Produce theoretical observation
    # with error corruption
    err = np.random.uniform(-1, 1, gr.outputs.shape[0]) * 1e-3
    obs = gr.outputs[:,0].copy()+err

    # Calibrate
    final1, ofun1, _, _ = calib1.workflow(obs, inputs, \
                                maxfun=100000, ftol=1e-8)
    final2, ofun2, _, _ = calib2.workflow(obs, inputs, \
                                maxfun=100000, ftol=1e-8)

    assert allclose(final1, expected, atol=5e-1)
    assert allclose(final2[0], 200)
