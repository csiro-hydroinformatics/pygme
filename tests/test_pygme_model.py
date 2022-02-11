import os
import re
import pytest

from timeit import Timer
import time

import numpy as np
np.seterr(all="print")

from hydrodiy.data.containers import Vector
from pygme.model import Model, NORDMAXMAX, UH, UHNAMES, ParamsVector, \
                            ParameterCheckValueError
from dummy import Dummy, MassiveDummy


def test_init():
    for nm in UHNAMES:
        u = UH(nm)
        s = str(u)

    msg = "Expected UH name"
    with pytest.raises(ValueError, match=msg):
        u = UH("pignouf")


def test_set_timebase(allclose):
    for nm in UHNAMES:
        u = UH(nm)
        for p in np.linspace(0, 100, 500):
            u.timebase = p
            assert allclose(np.sum(u.ord[:u.nord]), 1., atol=1e-8)
            assert allclose(u.states, 0.)


def test_set_states(allclose):
    for nm in UHNAMES:
        u = UH(nm)
        u.states[:10] = 10.
        u.timebase = 20
        assert allclose(u.states[:u.nord], 0.)


def test_reset(allclose):
    for nm in UHNAMES:
        u = UH(nm)
        u.timebase = 5.5
        u.states = np.random.uniform(size=u.nord)
        u.reset()
        assert allclose(u.states, np.zeros(u.nordmax))


def test_initialise(allclose):
    for nm in UHNAMES:
        u = UH(nm)
        u.timebase = 5.5
        nord = u.nord

        states = np.random.uniform(size=nord)
        u.states = states
        assert allclose(u.states[:nord], states[:nord])

        msg = "Expected state vector"
        with pytest.raises(ValueError, match=msg):
            u.states = [0., 10.]

def test_uh_nordmax():
    u = UH("lag", nordmax=5)
    msg = "When setting param to"
    with pytest.raises(ValueError, match=msg):
        u.timebase = 10


def test_uh_lag(allclose):
    u = UH("lag")

    u.timebase = 5.5
    o = np.zeros(u.nordmax)
    o[5:7] = 0.5
    assert allclose(u.ord, o)

    u.timebase = 5
    o = np.zeros(u.nordmax)
    o[5] = 1
    assert allclose(u.ord, o)


def test_uh_triangle(allclose):
    u = UH("triangle")

    u.timebase = 2.5
    o = [0.08, 0.24, 0.36, 0.24, 0.08, 0.]
    assert allclose(u.ord[:len(o)], o)

    u.timebase = 9.2
    o = [0.005907372, 0.017722117, 0.029536862, 0.041351607, 0.053166352, \
            0.064981096, 0.076795841, 0.088610586, 0.100425331, \
            0.104678639, 0.093336484, 0.081521739, 0.069706994, \
            0.057892250, 0.046077505, 0.034262760, 0.022448015, \
            0.010633270, 0.000945180, 0.000000000]
    assert allclose(u.ord[:len(o)], o)


def test_uh_flat(allclose):
    u = UH("flat")

    u.timebase = 2.5
    o = [0.4, 0.4, 0.2]
    assert allclose(u.ord[:len(o)], o)

    u.timebase = 9.2
    o = [0.108695652]*9+[0.021739130, 0.]
    assert allclose(u.ord[:len(o)], o)


def test_uh_gr4j_ssh1_daily(allclose):
    u = UH("gr4j_ss1_daily")

    u.timebase = 2.5
    o = [0.10119288512539, 0.47124051711456, 0.42756659776005, 0.]
    assert allclose(u.ord[:len(o)], o)

    u.timebase = 1.3
    o = [0.51896924219351, 0.48103075780649, 0.]
    assert allclose(u.ord[:len(o)], o)


def test_uh_gr4j_ssh2_daily(allclose):
    u = UH("gr4j_ss2_daily")
    u.timebase = 2.5

    o = [0.05059644256269, 0.23562025855728, 0.42756659776005, \
                        0.23562025855728, 0.05059644256269]
    assert allclose(u.ord[:len(o)], o)

    u.timebase = 1.3
    o = [0.25948462109675, 0.66815684654371, 0.07235853235954, 0.]
    assert allclose(u.ord[:len(o)], o)



def test_init(allclose):
    vect = Vector(["X{0}".format(k) for k in range(10)])
    pv = ParamsVector(vect)
    pv.add_uh("lag", lambda params: params.X3)
    pv.add_uh("lag", lambda params: params.X6)
    pv.add_uh("lag", lambda params: params.X8)

    for k in range(len(pv.uhs)):
        uh = pv.uhs[k][1]
        assert allclose(uh.timebase, 0.)
        assert allclose(uh.nord, 1)

        ordi = np.zeros(uh.nordmax)
        ordi[0] = 1
        assert allclose(uh.ord, ordi)

        states = np.zeros(uh.nordmax)
        assert allclose(uh.states, states)


def test_error_init():
    vect = Vector(["X{0}".format(k) for k in range(10)])
    pv = ParamsVector(vect)
    msg = "Expected set_timebase "+\
            "function to return a float"
    with pytest.raises(ValueError, match=msg):
        pv.add_uh("lag", lambda params: [params.X1]*3)


def test_set_params(allclose):
    vect = Vector(["X{0}".format(k) for k in range(10)])
    pv = ParamsVector(vect)
    pv.add_uh("lag", lambda params: params.X3)
    pv.add_uh("lag", lambda params: params.X6)
    pv.add_uh("lag", lambda params: params.X8)

    # Set params
    pv.X3 = 10
    pv.X6 = 2.5
    pv.X8 = 5

    # Run comparison
    zero = np.zeros(pv.uhs[0][1].nordmax)
    o = zero.copy()
    o[10] = 1
    assert allclose(pv.uhs[0][1].ord, o)

    o = zero.copy()
    o[2:4] = 0.5
    assert allclose(pv.uhs[1][1].ord, o)


def test_set_params_complex(allclose):
    vect = Vector(["X{0}".format(k) for k in range(10)])
    pv = ParamsVector(vect)
    pv.add_uh("lag", lambda params: params.X1+params.X3*10)

    # Set params
    pv.X1 = 10
    pv.X3 = 2.5

    # Run comparison
    assert allclose(pv.uhs[0][1].timebase, 35)


def test_checkvalues():
    vect = Vector(["X{0}".format(k) for k in range(10)])
    def fun(values):
        if np.any(values < -10):
            raise ParameterCheckValueError("One parameter value"+\
                            " is < -10")

    pv = ParamsVector(vect, checkvalues=fun)

    # Set params (no error)
    pv.X1 = 10
    pv.X3 = 2.5

    # Set params (error)
    msg = "One parameter"
    with pytest.raises(Exception, match=msg):
        pv.X1 = -11

    # Set params (no error)
    values = np.zeros(pv.nval)
    pv.values = values

    values[0] = -11.
    msg = "One parameter"
    with pytest.raises(Exception, match=msg):
        pv.values = values

    # Problem with checkvalues function
    def fun(values):
        pass

    msg = "checkvalues function"
    with pytest.raises(Exception, match=msg):
        pv = ParamsVector(vect, checkvalues=fun)


def test_print():
    dum = Dummy()
    str = "{0}".format(dum)


def test_outputs_names():
    dum = Dummy()
    assert dum.outputs_names == ["a", "b"]

    dum.outputs_names = ["d", "e"]

    msg = "model dummy: Trying"
    with pytest.raises(ValueError, match=msg):
        dum.outputs_names = ["a", "b", "c"]


def test_allocate():
    dum = Dummy()
    inputs = np.random.uniform(0, 1, (1000, 2))

    msg = "Trying to get ntimesteps"
    with pytest.raises(ValueError, match=msg):
        n = dum.ntimesteps

    dum.allocate(inputs)
    assert dum.ninputs == 2
    assert dum.ntimesteps == 1000
    assert dum.istart == 0
    assert dum.iend == 999
    assert dum.inputs.shape == (1000, 2)


def test_set_params(allclose):
    params = [0.5, 10.]
    dum = Dummy()
    dum.params.values = params
    assert allclose(dum.params.values, params)


def test_set_params_attributes(allclose):
    params = [0.5, 10.]
    dum = Dummy()
    dum.X1 = params[0]
    dum.X2 = params[1]
    assert allclose(dum.params.values, params)


def test_set_get_items():
    pass


def test_initialise_states(allclose):
    dum = Dummy()
    states = [5, 6]
    dum.initialise(states)
    assert allclose(dum.states.values, states)


def test_initialise_uh(allclose):
    dum = Dummy()
    uh = UH(dum.params.uhs[0][1].name)
    uh.timebase = dum.params.uhs[0][1].timebase
    uh.states += 4.

    # Add the uh with no set_timebase function
    dum.initialise(uhs=[(None, uh)])

    assert allclose(dum.params.uhs[0][1].states, uh.states)


def test_set_inputs(allclose):
    nval = 100
    inputs1 = np.random.uniform(0, 1, (nval, 2))
    inputs2 = np.random.uniform(0, 1, (nval, 2))
    dum = Dummy()
    dum.inputs = inputs1
    assert dum.ntimesteps == nval
    assert allclose(dum.inputs, inputs1)

    dum.inputs = inputs2
    assert dum.ntimesteps == nval
    assert allclose(dum.inputs, inputs2)

    # Change inputs
    nval = 10
    inputs3 = np.random.uniform(0, 1, (nval, 2))
    dum.inputs = inputs3
    assert dum.ntimesteps == nval


def test_run(allclose):
    nval = 100
    inputs = np.random.uniform(0, 1, (nval, 2))
    dum = Dummy()
    dum.allocate(inputs, 2)

    params = [0.5, 10.]
    dum.params.values = params
    dum.config["continuous"] = 1

    states = np.array([10., 0.])
    dum.initialise(states=states)
    dum.run()

    expected = params[0] + params[1] * inputs
    expected = expected + states[:2][None, :]
    assert allclose(expected, dum.outputs[:, :2])


def test_inputs(allclose):
    inputs = np.random.uniform(0, 1, (1000, 2))

    dum = Dummy()
    dum.allocate(inputs)

    params = [0.5, 10.]
    dum.params.values = params
    dum.initialise(states=[10, 5])
    dum.config.values = [10]

    dum2 = dum.clone()
    dum2.allocate(inputs)

    d1 = dum.inputs
    d2 = dum2.inputs
    assert allclose(d1, d2)

    # Check that inputs were copied and not pointing to same object
    d2[0, 0] += 1
    assert allclose(d1[0, 0]+1, d2[0, 0])


def test_uh(allclose):
    dum = Dummy()
    inputs = np.random.uniform(0, 1, (10, 2))
    dum.allocate(inputs, 2)

    dum.params.values = np.array([4, 0.])
    nval = dum.params.uhs[0][1].ord.shape[0]
    o = np.array([0.25]*4 + [0.] * (nval-4))
    assert allclose(dum.params.uhs[0][1].ord, o)

    dum.params["X1"] = 6
    o = np.array([1./6]*6 + [0.] * (nval-6))
    assert allclose(dum.params.uhs[0][1].ord, o)


def test_run_default():
    dum = MassiveDummy()
    dum.params.values = 0.
    inputs = np.random.uniform(0, 1, (1000, 2))
    dum.allocate(inputs)
    dum.initialise()
    dum.run()


def test_allocate_dummy():
    dum = Dummy()
    nval = 1000
    ninputs = 2
    dum.allocate(np.random.uniform(0, 1, (nval, ninputs)))
    nts = dum.ntimesteps

    assert dum.params.nval == 2
    assert dum.states.nval == 2
    assert dum.params.uhs[0][1].ord.shape[0] == NORDMAXMAX
    assert dum.inputs.shape  == (nts, 2)
    assert dum.outputs.shape  == (nts, 1)


def test_run_startend(allclose):
    dum = Dummy()
    nval = 1000
    ninputs = 2
    inputs = np.random.uniform(0, 1, (nval, ninputs))
    dum.allocate(inputs)
    dum.params.value = [1., 2., 0.]

    dum.istart = 10
    dum.run()
    assert np.all(np.isnan(dum.outputs[:dum.istart, 0]))

    msg = "model dummy: Expected iend in \[0, 999\], got 1001"
    with pytest.raises(ValueError, match=msg):
        dum.iend = nval+1


def test_inisens():
    """ Test sensitivity to initial conditions """
    dum = Dummy()
    dum.config.continuous = 1
    nval = 1000
    ninputs = 2
    inputs = np.random.uniform(0, 1, (nval, ninputs))
    dum.allocate(inputs)
    dum.params.value = [1., 2., 0.]

    msg = "Warmup period"
    with pytest.raises(ValueError, match=msg):
        warmup, sim0, sim1 = dum.inisens([0]*2, [1]*2)


