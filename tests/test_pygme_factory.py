import re
from itertools import product as prod
import pytest

from pygme.factory import MODEL_NAMES, model_factory, calibration_factory
from pygme.factory import parameters_transform_factory, objfun_factory

from pygme.calibration import ObjFunBCSSE


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_model_factory(model_name):
    model = model_factory(model_name)


def test_model_factory_errors():
    msg = f"Expected model name in "
    with pytest.raises(ValueError, match=msg):
        model = model_factory("bidule")


@pytest.mark.parametrize("shape", [0, 1, 2, 2.5])
def test_model_factory_ihacres(shape):
    model = model_factory(f"IHACRES{shape}")
    assert model.shapefactor == shape


@pytest.mark.parametrize("rcap", [10, 60, 1000])
def test_model_factory_gr2m(rcap):
    model = model_factory(f"GR2M{rcap}")
    assert model.Rcapacity == rcap


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_calibration_factory(model_name):
    model = calibration_factory(model_name)


def test_calibration_factory_errors():
    msg = f"Expected model name in "
    with pytest.raises(ValueError, match=msg):
        calib = calibration_factory("bidule")


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_calibration_factory_argument(model_name):
    objfun = ObjFunBCSSE(0.2)
    calib = calibration_factory(model_name, objfun=objfun)
    assert hasattr(calib.objfun, "trans")
    assert calib.objfun.trans.lam == 0.2


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_paramtrans_factory(model_name):
    if model_name != "TurcMezentsev":
        f1, f2 = parameters_transform_factory(model_name)
    else:
        msg = "No transform available"
        with pytest.raises(ValueError, match=msg):
            f1, f2 = parameters_transform_factory(model_name)

@pytest.mark.parametrize("name", ["bc0.0", "bc1.0", "biasbc0.0",
                                  "bc0.0_1", "bc1.0_5.0",
                                  "biasbc1.0", "sse", "kge"])
def test_objfun_factory(name):
    objfun = objfun_factory(name)
    if re.search("bc", name):
        lam = float(re.sub(".*bc|_.*$", "", name))
        assert objfun.trans.lam == lam

        if re.search("bc.*_", name):
            nu = float(re.sub(".*_", "", name))
            assert objfun.trans.nu == nu
        else:
            assert objfun.trans.nu == 1e-10


def test_objfun_factory_errors():
    msg = f"Objective function truc not recognised"
    with pytest.raises(ValueError, match=msg):
        objfun = objfun_factory("truc")
