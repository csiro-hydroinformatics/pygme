import re
from itertools import product as prod
import pytest

from pygme.factory import MODEL_NAMES, model_factory, calibration_factory, \
                            parameters_transform_factory, \
                            objfun_factory

from pygme.calibration import ObjFunBCSSE


def test_model_factory():
    for model_name in MODEL_NAMES:
        model = model_factory(model_name)


def test_model_factory_errors():
    msg = f"Expected model name in "
    with pytest.raises(ValueError, match=msg):
        model = model_factory("bidule")


def test_model_factory_ihacres():
    for shape in [0, 1, 2, 2.5]:
        model = model_factory(f"IHACRES{shape}")
        assert model.shapefactor == shape


def test_model_factory_gr2m():
    for r in [10, 60, 1000]:
        model = model_factory(f"GR2M{r}")
        assert model.Rcapacity == r


def test_calibration_factory():
    for model_name in MODEL_NAMES:
        model = calibration_factory(model_name)


def test_calibration_factory_errors():
    msg = f"Expected model name in "
    with pytest.raises(ValueError, match=msg):
        calib = calibration_factory("bidule")


def test_calibration_factory_ihacres():
    for shape in [0, 1, 2, 2.5]:
        calib = calibration_factory(f"IHACRES{shape}")
        assert calib.model.shapefactor == shape


def test_calibration_factory_gr2m():
    for r in [10, 60, 1000]:
        calib = calibration_factory(f"GR2M{r}")
        assert calib.model.Rcapacity == r


def test_calibration_factory_argument():
    objfun = ObjFunBCSSE(0.2)
    for model_name in MODEL_NAMES:
        calib = calibration_factory(model_name, objfun=objfun)
        assert hasattr(calib.objfun, "trans")
        assert calib.objfun.trans.lam == 0.2


def test_paramtrans_factory():
    for model_name in MODEL_NAMES:
        if model_name != "TurcMezentsev":
            f1, f2 = parameters_transform_factory(model_name)
        else:
            msg = "No transform available"
            with pytest.raises(ValueError, match=msg):
                f1, f2 = parameters_transform_factory(model_name)

def test_objfun_factory():
    for name in ["bc0.0", "bc1.0", "biasbc0.0", \
            "bc0.0_1", "bc1.0_5.0", \
            "biasbc1.0", "sse", "kge"]:
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
